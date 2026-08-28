#!/usr/bin/env python3
"""Export the public ATST-F strong checkpoint as a TensorRT engine.

The engine accepts the normalized ATST mel tensor [1, 1, 64, 1001] and emits
[1, 447, 250] probabilities. Audio decoding, mel extraction, and the editable
class_mapping.csv aggregation intentionally live in the C++ worker.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.atstframe.ATSTF_wrapper import ATSTWrapper  # noqa: E402
from models.prediction_wrapper import PredictionsWrapper  # noqa: E402
from data_util.audioset_classes import as_strong_train_classes  # noqa: E402


class StrongAtst(nn.Module):
    """ATST strong inference returning all editable source probabilities."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.model = PredictionsWrapper(
            ATSTWrapper(),
            checkpoint=None,
            n_classes_strong=447,
            seq_model_type=None,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        allowed_missing = {
            "model.atst_mel.mel_transform.spectrogram.window",
            "model.atst_mel.mel_transform.mel_scale.fb",
        }
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Unexpected checkpoint mismatch; missing={missing}, unexpected={unexpected}"
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(mel)
        return torch.sigmoid(logits).reshape(1, 447, 250)


def find_trtexec(explicit: Path | None) -> Path | None:
    if explicit is not None:
        if explicit.is_file():
            return explicit.resolve()
        print(
            f"Warning: trtexec does not exist at {explicit}; "
            "trying the TensorRT Python Builder API"
        )
        return None
    resolved = shutil.which("trtexec")
    if resolved:
        return Path(resolved)
    jetson_path = Path("/usr/src/tensorrt/bin/trtexec")
    if jetson_path.is_file():
        return jetson_path
    return None


def build_with_tensorrt_python(
    onnx_path: Path,
    output: Path,
    workspace_mib: int,
    fp32: bool,
) -> None:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "Neither trtexec nor the Python 'tensorrt' module is installed. "
            "Install the TensorRT development/runtime packages supplied by JetPack."
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_mib * 1024 * 1024,
    )
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3
    if not fp32:
        config.set_flag(trt.BuilderFlag.FP16)
    print(f"Building TensorRT engine with Python TensorRT {trt.__version__}: {output}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT Python builder failed to create a serialized engine")
    output.write_bytes(bytes(serialized))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "resources" / "ATST-F_strong_1.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "resources" / "ATST-F_strong_1.trt",
    )
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=None,
        help="Ordered label file for the C++ worker (default: OUTPUT.labels.txt).",
    )
    parser.add_argument("--trtexec", type=Path, default=None)
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument(
        "--engine-only",
        action="store_true",
        help="Reuse existing ONNX and labels files and only build the TensorRT engine.",
    )
    parser.add_argument("--fp32", action="store_true", help="Build FP32 instead of FP16.")
    parser.add_argument("--workspace-mib", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    onnx_path = (args.onnx or output.with_suffix(".onnx")).resolve()
    labels_path = (args.labels_output or output.with_suffix(".labels.txt")).resolve()
    if not args.engine_only and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    if args.engine_only:
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Existing ONNX file not found: {onnx_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Existing ordered labels file not found: {labels_path}")
        print(f"Reusing ONNX: {onnx_path}")
        print(f"Reusing ordered labels: {labels_path}")
    else:
        model = StrongAtst(checkpoint).eval()
        example = torch.zeros(1, 1, 64, 1001, dtype=torch.float32)
        with torch.inference_mode():
            result = model(example)
        if tuple(result.shape) != (1, 447, 250):
            raise RuntimeError(f"Unexpected PyTorch output shape: {tuple(result.shape)}")
        if len(as_strong_train_classes) != 447:
            raise RuntimeError("Expected exactly 447 ordered ATST-F labels")
        labels_path.write_text("\n".join(as_strong_train_classes) + "\n", encoding="utf-8")
        print(f"Wrote ordered labels: {labels_path}")

        print(f"Exporting ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            example,
            str(onnx_path),
            input_names=["mel"],
            output_names=["probabilities"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        try:
            import onnx

            graph = onnx.load(str(onnx_path))
            onnx.checker.check_model(graph)
            print("ONNX validation passed")
        except ImportError:
            print("Warning: Python package 'onnx' is unavailable; skipping ONNX checker")

    if args.onnx_only:
        return

    trtexec = find_trtexec(args.trtexec)
    if trtexec is not None:
        command = [
            str(trtexec),
            f"--onnx={onnx_path}",
            f"--saveEngine={output}",
            f"--memPoolSize=workspace:{args.workspace_mib}",
            "--builderOptimizationLevel=3",
            "--skipInference",
        ]
        if not args.fp32:
            command.append("--fp16")
        print("Building TensorRT engine:", " ".join(command))
        subprocess.run(command, check=True)
    else:
        build_with_tensorrt_python(
            onnx_path,
            output,
            args.workspace_mib,
            args.fp32,
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"TensorRT engine was not created: {output}")
    print(f"TensorRT engine created: {output}")


if __name__ == "__main__":
    main()
