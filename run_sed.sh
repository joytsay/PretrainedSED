#!/usr/bin/env bash
set -euo pipefail

container_name="${PSED_CONTAINER_NAME:-psed}"
image_name="${PSED_IMAGE:-psed:latest}"
host_port="${PSED_PORT:-8080}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ((EUID == 0)); then
  docker_command=(docker)
else
  docker_command=(sudo docker)
fi

usage() {
  cat <<'EOF'
Usage: ./run_sed.sh [--recreate]

Starts the SED web testbed in a Docker container. The container uses the
"unless-stopped" restart policy, so Docker starts it again after host reboots.

Options:
  --recreate  Replace an existing container to apply a new image or launch config.

Environment overrides:
  PSED_CONTAINER_NAME  Container name (default: psed)
  PSED_IMAGE           Docker image (default: psed:latest)
  PSED_PORT            Host HTTP port (default: 8080)
EOF
}

recreate=false
case "${1:-}" in
  "") ;;
  --recreate) recreate=true ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

required_files=(
  "build-agx/atst_sed_web_testbed"
  "build-agx/atst_sed_worker"
  "resources/ATST-F_strong_1.trt"
  "resources/ATST-F_strong_1.labels.txt"
  "class_mapping.csv"
  "src/webui/dist/index.html"
)

for relative_path in "${required_files[@]}"; do
  if [[ ! -e "${script_dir}/${relative_path}" ]]; then
    printf 'Missing required file: %s\n' "${script_dir}/${relative_path}" >&2
    exit 1
  fi
done

if ! "${docker_command[@]}" info >/dev/null 2>&1; then
  printf 'Docker is unavailable. Ensure the service is running and try again.\n' >&2
  exit 1
fi

if "${docker_command[@]}" container inspect "${container_name}" >/dev/null 2>&1; then
  if [[ "${recreate}" == true ]]; then
    "${docker_command[@]}" container rm --force "${container_name}" >/dev/null
  else
    "${docker_command[@]}" update --restart unless-stopped "${container_name}" >/dev/null
    if [[ "$("${docker_command[@]}" inspect --format '{{.State.Running}}' "${container_name}")" != true ]]; then
      "${docker_command[@]}" start "${container_name}" >/dev/null
    fi
    printf 'SED container %s is running at http://0.0.0.0:%s\n' \
      "${container_name}" "${host_port}"
    printf 'Use %s --recreate to apply image or launch-command changes.\n' "$0"
    exit 0
  fi
fi

"${docker_command[@]}" run -d \
  --name "${container_name}" \
  --restart unless-stopped \
  --init \
  --runtime nvidia \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "${host_port}:8080" \
  -v "${script_dir}:/workspace" \
  -w /workspace \
  "${image_name}" \
  bash -lc '
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}"
    exec /workspace/build-agx/atst_sed_web_testbed \
      --worker /workspace/build-agx/atst_sed_worker \
      --engine /workspace/resources/ATST-F_strong_1.trt \
      --mapping /workspace/class_mapping.csv \
      --labels /workspace/resources/ATST-F_strong_1.labels.txt \
      --web-root /workspace/src/webui/dist \
      --videos-root /workspace/videos \
      --host 0.0.0.0 \
      --port 8080
  ' >/dev/null

printf 'SED container %s started at http://0.0.0.0:%s\n' \
  "${container_name}" "${host_port}"
printf 'Follow logs with: %s logs --follow %s\n' \
  "${docker_command[*]}" "${container_name}"
