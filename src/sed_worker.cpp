#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <deque>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>

namespace {

using json = nlohmann::json;
constexpr int kSampleRate = 16000;
constexpr int kChunkSamples = 160000;
constexpr int kWindowMilliseconds = 10000;
constexpr int kPacketMilliseconds = 40;
constexpr int kPacketSamples = kSampleRate * kPacketMilliseconds / 1000;
constexpr int kFftSize = 1024;
constexpr int kHopSize = 160;
constexpr int kMelBins = 64;
constexpr int kMelFrames = 1001;
constexpr int kModelClasses = 447;
constexpr int kAggregateClasses = 3;
constexpr int kOutputFrames = 250;
constexpr double kPi = 3.14159265358979323846;

void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorRT: " << message << '\n';
        }
    }
};

template <typename T>
struct TrtDelete {
    void operator()(T* object) const { delete object; }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

std::vector<char> readBinary(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open TensorRT engine: " + path);
    const auto size = stream.tellg();
    if (size <= 0) throw std::runtime_error("TensorRT engine is empty: " + path);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    stream.read(bytes.data(), size);
    if (!stream) throw std::runtime_error("Could not read TensorRT engine: " + path);
    return bytes;
}

std::int64_t volume(const nvinfer1::Dims& dims) {
    std::int64_t result = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) throw std::runtime_error("TensorRT returned a dynamic or invalid tensor dimension");
        result *= dims.d[index];
    }
    return result;
}

class TensorRtModel {
public:
    explicit TensorRtModel(const std::string& enginePath) {
        const auto bytes = readBinary(enginePath);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("Could not create TensorRT runtime");
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine_) throw std::runtime_error("Could not deserialize TensorRT engine: " + enginePath);
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Could not create TensorRT execution context");

        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                inputName_ = name;
            } else {
                outputName_ = name;
            }
        }
        if (inputName_.empty() || outputName_.empty()) {
            throw std::runtime_error("Expected one TensorRT input and one output");
        }
        if (engine_->getTensorDataType(inputName_.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType(outputName_.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Engine I/O tensors must be float32 (FP16 internal layers are supported)");
        }

        nvinfer1::Dims inputDims{};
        inputDims.nbDims = 4;
        inputDims.d[0] = 1;
        inputDims.d[1] = 1;
        inputDims.d[2] = kMelBins;
        inputDims.d[3] = kMelFrames;
        if (!context_->setInputShape(inputName_.c_str(), inputDims)) {
            throw std::runtime_error("Engine rejected input shape 1x1x64x1001");
        }
        const auto outputDims = context_->getTensorShape(outputName_.c_str());
        if (volume(outputDims) != kModelClasses * kOutputFrames) {
            throw std::runtime_error("Expected TensorRT output with 447x250 values");
        }

        cudaCheck(cudaMalloc(&deviceInput_, sizeof(float) * kMelBins * kMelFrames), "cudaMalloc input");
        cudaCheck(cudaMalloc(&deviceOutput_, sizeof(float) * kModelClasses * kOutputFrames), "cudaMalloc output");
        if (!context_->setTensorAddress(inputName_.c_str(), deviceInput_) ||
            !context_->setTensorAddress(outputName_.c_str(), deviceOutput_)) {
            throw std::runtime_error("Could not bind TensorRT tensor addresses");
        }
        cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~TensorRtModel() {
        if (stream_) cudaStreamDestroy(stream_);
        if (deviceOutput_) cudaFree(deviceOutput_);
        if (deviceInput_) cudaFree(deviceInput_);
    }

    std::vector<float> infer(const std::vector<float>& mel) {
        if (mel.size() != static_cast<std::size_t>(kMelBins * kMelFrames)) {
            throw std::runtime_error("Invalid mel input size");
        }
        std::vector<float> output(kModelClasses * kOutputFrames);
        cudaCheck(cudaMemcpyAsync(deviceInput_, mel.data(), mel.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_), "copy mel to GPU");
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
        cudaCheck(cudaMemcpyAsync(output.data(), deviceOutput_, output.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_), "copy scores from GPU");
        cudaCheck(cudaStreamSynchronize(stream_), "TensorRT synchronization");
        return output;
    }

private:
    Logger logger_;
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    std::string inputName_;
    std::string outputName_;
    void* deviceInput_ = nullptr;
    void* deviceOutput_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

struct ClassMapping {
    std::array<std::string, kAggregateClasses> names;
    std::array<std::vector<int>, kAggregateClasses> sourceIndices;
};

struct StreamState {
    std::string camId;
    std::deque<float> samples;
    std::int64_t nextTimestampMs = 0;
    std::int64_t lastTimestampMs = -1;
    std::uint64_t packetsReceived = 0;
};

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> parseCsvRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(character);
        }
    }
    if (quoted) throw std::runtime_error("Unterminated quoted field in class mapping CSV");
    fields.push_back(trim(field));
    return fields;
}

ClassMapping loadClassMapping(const std::string& mappingPath, const std::string& labelsPath) {
    std::ifstream labelsFile(labelsPath);
    if (!labelsFile) throw std::runtime_error("Cannot open ordered label file: " + labelsPath);
    std::unordered_map<std::string, int> labelIndices;
    std::string line;
    int labelIndex = 0;
    while (std::getline(labelsFile, line)) {
        const std::string label = trim(line);
        if (label.empty()) continue;
        if (!labelIndices.emplace(label, labelIndex).second) {
            throw std::runtime_error("Duplicate source label in ordered label file: " + label);
        }
        ++labelIndex;
    }
    if (labelIndex != kModelClasses) {
        throw std::runtime_error(
            "Ordered label file must contain exactly 447 labels; found " + std::to_string(labelIndex));
    }

    std::ifstream mappingFile(mappingPath);
    if (!mappingFile) throw std::runtime_error("Cannot open class mapping CSV: " + mappingPath);
    ClassMapping mapping;
    std::unordered_map<std::string, int> aggregateIndices;
    int lineNumber = 0;
    while (std::getline(mappingFile, line)) {
        ++lineNumber;
        const std::string stripped = trim(line);
        if (stripped.empty() || stripped[0] == '#') continue;
        const auto fields = parseCsvRow(line);
        if (fields.size() < 2) {
            throw std::runtime_error(
                "class_mapping.csv line " + std::to_string(lineNumber) + " needs two columns");
        }
        if (fields[0] == "class_name" && fields[1] == "source_class") continue;
        if (fields[0].empty() || fields[1].empty()) {
            throw std::runtime_error(
                "class_mapping.csv line " + std::to_string(lineNumber) + " has an empty value");
        }
        auto aggregate = aggregateIndices.find(fields[0]);
        int aggregateIndex = 0;
        if (aggregate == aggregateIndices.end()) {
            aggregateIndex = static_cast<int>(aggregateIndices.size());
            if (aggregateIndex >= kAggregateClasses) {
                throw std::runtime_error("The testbed requires exactly three aggregate classes");
            }
            aggregateIndices.emplace(fields[0], aggregateIndex);
            mapping.names[aggregateIndex] = fields[0];
        } else {
            aggregateIndex = aggregate->second;
        }
        const auto source = labelIndices.find(fields[1]);
        if (source == labelIndices.end()) {
            throw std::runtime_error(
                "Unknown source class on class_mapping.csv line " +
                std::to_string(lineNumber) + ": " + fields[1]);
        }
        auto& group = mapping.sourceIndices[aggregateIndex];
        if (std::find(group.begin(), group.end(), source->second) != group.end()) {
            throw std::runtime_error(
                "Duplicate source class in aggregate " + fields[0] + ": " + fields[1]);
        }
        group.push_back(source->second);
    }
    if (aggregateIndices.size() != kAggregateClasses) {
        throw std::runtime_error(
            "The testbed requires exactly three aggregate classes; found " +
            std::to_string(aggregateIndices.size()));
    }
    return mapping;
}

std::vector<std::uint8_t> decodeBase64(const std::string& encoded) {
    static const std::array<int, 256> table = [] {
        std::array<int, 256> values{};
        values.fill(-1);
        const std::string alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (std::size_t index = 0; index < alphabet.size(); ++index) {
            values[static_cast<unsigned char>(alphabet[index])] = static_cast<int>(index);
        }
        return values;
    }();

    std::vector<std::uint8_t> decoded;
    decoded.reserve(encoded.size() * 3 / 4);
    std::uint32_t accumulator = 0;
    int bits = -8;
    for (const unsigned char character : encoded) {
        if (character == '=') break;
        const int value = table[character];
        if (value < 0) throw std::runtime_error("audio_b64 contains invalid base64");
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<std::uint8_t>((accumulator >> bits) & 0xff));
            bits -= 8;
        }
    }
    return decoded;
}

std::vector<float> decodePcm16Packet(const json& request) {
    if (request.value("sample_rate", 0) != kSampleRate) {
        throw std::runtime_error("Audio packet sample_rate must be 16000");
    }
    if (request.value("channels", 0) != 1) {
        throw std::runtime_error("Audio packet channels must be 1");
    }
    if (request.value("encoding", std::string()) != "s16le") {
        throw std::runtime_error("Audio packet encoding must be s16le");
    }
    const auto bytes = decodeBase64(request.at("audio_b64").get<std::string>());
    if (bytes.size() != static_cast<std::size_t>(kPacketSamples * 2)) {
        throw std::runtime_error(
            "Each audio packet must contain exactly 640 mono s16le samples (1280 bytes)");
    }

    std::vector<float> samples(kPacketSamples);
    for (int index = 0; index < kPacketSamples; ++index) {
        int value = static_cast<int>(bytes[index * 2]) |
                    (static_cast<int>(bytes[index * 2 + 1]) << 8);
        if (value >= 32768) value -= 65536;
        samples[index] = static_cast<float>(value) / 32768.0F;
    }
    return samples;
}

void fft(std::vector<std::complex<float>>& values) {
    const std::size_t count = values.size();
    for (std::size_t i = 1, j = 0; i < count; ++i) {
        std::size_t bit = count >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }
    for (std::size_t length = 2; length <= count; length <<= 1) {
        const float angle = static_cast<float>(-2.0 * kPi / static_cast<double>(length));
        const std::complex<float> base(std::cos(angle), std::sin(angle));
        for (std::size_t offset = 0; offset < count; offset += length) {
            std::complex<float> factor(1.0F, 0.0F);
            for (std::size_t index = 0; index < length / 2; ++index) {
                const auto even = values[offset + index];
                const auto odd = values[offset + index + length / 2] * factor;
                values[offset + index] = even + odd;
                values[offset + index + length / 2] = even - odd;
                factor *= base;
            }
        }
    }
}

float reflectSample(const std::vector<float>& samples, int index) {
    const int size = static_cast<int>(samples.size());
    if (size == 1) return samples[0];
    while (index < 0 || index >= size) {
        if (index < 0) index = -index;
        if (index >= size) index = 2 * size - index - 2;
    }
    return samples[index];
}

float hzToMel(float hz) { return 2595.0F * std::log10(1.0F + hz / 700.0F); }
float melToHz(float mel) { return 700.0F * (std::pow(10.0F, mel / 2595.0F) - 1.0F); }

std::vector<float> makeMelFilter() {
    constexpr int frequencyBins = kFftSize / 2 + 1;
    std::array<float, kMelBins + 2> points{};
    const float minMel = hzToMel(60.0F);
    const float maxMel = hzToMel(7800.0F);
    for (int index = 0; index < kMelBins + 2; ++index) {
        points[index] = melToHz(minMel + (maxMel - minMel) * index / (kMelBins + 1));
    }
    std::vector<float> filter(kMelBins * frequencyBins, 0.0F);
    for (int mel = 0; mel < kMelBins; ++mel) {
        for (int bin = 0; bin < frequencyBins; ++bin) {
            const float hz = static_cast<float>(bin * kSampleRate) / kFftSize;
            const float lower = (hz - points[mel]) / (points[mel + 1] - points[mel]);
            const float upper = (points[mel + 2] - hz) / (points[mel + 2] - points[mel + 1]);
            filter[mel * frequencyBins + bin] = std::max(0.0F, std::min(lower, upper));
        }
    }
    return filter;
}

std::vector<float> computeMel(const std::vector<float>& chunk) {
    constexpr int frequencyBins = kFftSize / 2 + 1;
    static const std::vector<float> filter = makeMelFilter();
    std::array<float, kFftSize> window{};
    for (int index = 0; index < kFftSize; ++index) {
        window[index] = 0.5F - 0.5F * std::cos(static_cast<float>(2.0 * kPi * index / kFftSize));
    }
    std::vector<float> mel(kMelBins * kMelFrames, 0.0F);
    std::vector<std::complex<float>> spectrum(kFftSize);
    for (int frame = 0; frame < kMelFrames; ++frame) {
        const int start = frame * kHopSize - kFftSize / 2;
        for (int index = 0; index < kFftSize; ++index) {
            spectrum[index] = {reflectSample(chunk, start + index) * window[index], 0.0F};
        }
        fft(spectrum);
        for (int melBin = 0; melBin < kMelBins; ++melBin) {
            float power = 0.0F;
            for (int frequency = 0; frequency < frequencyBins; ++frequency) {
                power += std::norm(spectrum[frequency]) * filter[melBin * frequencyBins + frequency];
            }
            mel[melBin * kMelFrames + frame] = 10.0F * std::log10(std::max(power, 1.0e-10F));
        }
    }
    const float maximum = *std::max_element(mel.begin(), mel.end());
    for (float& value : mel) {
        value = std::max(value, maximum - 80.0F);
        value = std::clamp(value, -50.0F, 80.0F);
        value = (value - (-79.6482F)) / (50.6842F - (-79.6482F)) * 2.0F - 1.0F;
    }
    return mel;
}

json aggregateFrame(
    const std::vector<float>& output,
    const ClassMapping& mapping,
    int frame
) {
    json scores = json::array();
    for (const auto& sourceIndices : mapping.sourceIndices) {
        float score = 0.0F;
        for (const int sourceIndex : sourceIndices) {
            score += output[sourceIndex * kOutputFrames + frame];
        }
        scores.push_back(std::min(score, 1.0F));
    }
    return scores;
}

void callback(const json& message) {
    static std::mutex outputMutex;
    const std::lock_guard<std::mutex> lock(outputMutex);
    std::cout << message.dump() << '\n' << std::flush;
}

struct InferenceJob {
    std::int64_t id = -1;
    std::string camId;
    std::int64_t timestampMs = 0;
    std::vector<float> window;
    std::uint64_t supersededPackets = 0;
};

class InferenceScheduler {
public:
    InferenceScheduler(TensorRtModel& model, const ClassMapping& mapping)
        : model_(model), mapping_(mapping), thread_([this] { run(); }) {}

    ~InferenceScheduler() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_one();
        thread_.join();
    }

    void activate(std::int64_t id) {
        const std::lock_guard<std::mutex> lock(mutex_);
        activeStreams_.insert(id);
        pending_.erase(
            std::remove_if(pending_.begin(), pending_.end(),
                           [id](const InferenceJob& job) { return job.id == id; }),
            pending_.end());
    }

    void deactivate(std::int64_t id) {
        const std::lock_guard<std::mutex> lock(mutex_);
        activeStreams_.erase(id);
        pending_.erase(
            std::remove_if(pending_.begin(), pending_.end(),
                           [id](const InferenceJob& job) { return job.id == id; }),
            pending_.end());
    }

    void submit(InferenceJob job) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (activeStreams_.count(job.id) == 0) return;
            const auto existing = std::find_if(
                pending_.begin(), pending_.end(),
                [&job](const InferenceJob& pending) { return pending.id == job.id; });
            if (existing == pending_.end()) {
                pending_.push_back(std::move(job));
            } else {
                job.supersededPackets = existing->supersededPackets + 1;
                *existing = std::move(job);
            }
        }
        condition_.notify_one();
    }

private:
    void run() {
        while (true) {
            InferenceJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
                if (stopping_ && pending_.empty()) return;
                job = std::move(pending_.front());
                pending_.pop_front();
            }

            const auto started = std::chrono::steady_clock::now();
            const auto output = model_.infer(computeMel(job.window));
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started);

            const json result = {
                {"event", "result"},
                {"id", job.id},
                {"cam_id", job.camId},
                {"timestamp_ms", job.timestampMs},
                {"window_start_ms", job.timestampMs + kPacketMilliseconds - kWindowMilliseconds},
                {"window_end_ms", job.timestampMs + kPacketMilliseconds},
                {"processing_ms", elapsed.count()},
                {"superseded_packets", job.supersededPackets},
                {"scores", aggregateFrame(output, mapping_, kOutputFrames - 1)},
            };
            const std::lock_guard<std::mutex> lock(mutex_);
            if (activeStreams_.count(job.id) != 0) callback(result);
        }
    }

    TensorRtModel& model_;
    const ClassMapping& mapping_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<InferenceJob> pending_;
    std::unordered_set<std::int64_t> activeStreams_;
    bool stopping_ = false;
    std::thread thread_;
};

void processMessage(
    InferenceScheduler& scheduler,
    std::unordered_map<std::int64_t, StreamState>& streams,
    const json& request
) {
    const std::string type = request.at("type").get<std::string>();
    const std::int64_t id = request.at("id").get<std::int64_t>();
    const std::string camId = request.at("cam_id").get<std::string>();
    if (camId.empty()) throw std::runtime_error("cam_id must not be empty");

    if (type == "stream_start") {
        const std::int64_t startTimestampMs = request.at("timestamp_ms").get<std::int64_t>();
        if (startTimestampMs < 0) throw std::runtime_error("timestamp_ms must not be negative");
        StreamState stream;
        stream.camId = camId;
        stream.nextTimestampMs = startTimestampMs;
        stream.samples.assign(kChunkSamples, 0.0F);
        streams[id] = std::move(stream);
        scheduler.activate(id);
        callback({
            {"event", "stream_started"},
            {"id", id},
            {"cam_id", camId},
            {"packet_ms", kPacketMilliseconds},
            {"window_ms", kWindowMilliseconds},
            {"silence_prefill_ms", kWindowMilliseconds},
        });
        return;
    }

    const auto streamIterator = streams.find(id);
    if (streamIterator == streams.end()) {
        throw std::runtime_error("Unknown stream id; send stream_start before audio packets");
    }
    StreamState& stream = streamIterator->second;
    if (stream.camId != camId) throw std::runtime_error("cam_id changed within a stream");

    if (type == "stream_end") {
        scheduler.deactivate(id);
        callback({
            {"event", "complete"},
            {"id", id},
            {"cam_id", camId},
            {"packets_received", stream.packetsReceived},
            {"last_timestamp_ms", stream.lastTimestampMs},
        });
        streams.erase(streamIterator);
        return;
    }
    if (type != "audio") throw std::runtime_error("Unknown message type: " + type);

    const std::int64_t timestampMs = request.at("timestamp_ms").get<std::int64_t>();
    if (timestampMs < 0) throw std::runtime_error("timestamp_ms must not be negative");
    if (timestampMs != stream.nextTimestampMs) {
        throw std::runtime_error("Audio packet timestamps must increase by exactly 40 ms");
    }

    const auto packet = decodePcm16Packet(request);
    stream.samples.insert(stream.samples.end(), packet.begin(), packet.end());
    while (stream.samples.size() > static_cast<std::size_t>(kChunkSamples)) {
        stream.samples.pop_front();
    }
    stream.lastTimestampMs = timestampMs;
    stream.nextTimestampMs += kPacketMilliseconds;
    ++stream.packetsReceived;

    scheduler.submit({
        id,
        camId,
        timestampMs,
        std::vector<float>(stream.samples.begin(), stream.samples.end()),
        0,
    });
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: atst_sed_worker ENGINE.trt [class_mapping.csv] [ENGINE.labels.txt]\n";
        return 2;
    }
    try {
        const std::string enginePath = argv[1];
        const std::string mappingPath = argc >= 3 ? argv[2] : "class_mapping.csv";
        std::filesystem::path defaultLabels(enginePath);
        defaultLabels.replace_extension(".labels.txt");
        const std::string labelsPath = argc >= 4 ? argv[3] : defaultLabels.string();
        const ClassMapping mapping = loadClassMapping(mappingPath, labelsPath);
        TensorRtModel model(enginePath);
        InferenceScheduler scheduler(model, mapping);
        callback({
            {"event", "ready"},
            {"classes", {mapping.names[0], mapping.names[1], mapping.names[2]}},
            {"mapping", mappingPath},
            {"sample_rate", kSampleRate},
            {"channels", 1},
            {"encoding", "s16le"},
            {"packet_samples", kPacketSamples},
            {"packet_ms", kPacketMilliseconds},
            {"window_ms", kWindowMilliseconds},
            {"silence_prefill_ms", kWindowMilliseconds},
        });
        std::unordered_map<std::int64_t, StreamState> streams;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::int64_t id = -1;
            std::string camId;
            try {
                const auto request = json::parse(line);
                id = request.value("id", -1);
                camId = request.value("cam_id", std::string());
                processMessage(scheduler, streams, request);
            } catch (const std::exception& error) {
                callback({
                    {"event", "error"},
                    {"id", id},
                    {"cam_id", camId},
                    {"message", error.what()},
                });
            }
        }
    } catch (const std::exception& error) {
        callback({{"event", "fatal"}, {"message", error.what()}});
        return 1;
    }
    return 0;
}
