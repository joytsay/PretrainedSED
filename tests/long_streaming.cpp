#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr int kPacketMs = 40;
constexpr std::size_t kPacketBytes = 640 * 2;
volatile std::sig_atomic_t gStop = 0;
std::mutex gPrintMutex;
std::size_t gLiveRows = 0;

void onSignal(int) { gStop = 1; }

void log(const std::string& line) {
    std::lock_guard<std::mutex> lock(gPrintMutex);
    if (gLiveRows > 0) {
        if (gLiveRows > 1) std::cout << "\033[" << gLiveRows - 1 << 'B';
        std::cout << '\n';
        gLiveRows = 0;
    }
    std::cout << line << std::endl;
}

void showLivePanel(const std::vector<std::string>& lines) {
    std::lock_guard<std::mutex> lock(gPrintMutex);
    if (::isatty(STDOUT_FILENO)) {
        winsize size{};
        const std::size_t width = ::ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 1
                                      ? size.ws_col - 1 : 79;
        for (std::size_t index = 0; index < lines.size(); ++index) {
            std::cout << "\r\033[2K" << lines[index].substr(0, width);
            if (index + 1 < lines.size()) std::cout << '\n';
        }
        if (lines.size() > 1) std::cout << "\033[" << lines.size() - 1 << "A\r";
        std::cout << std::flush;
        gLiveRows = lines.size();
    } else {
        for (const auto& line : lines) std::cout << line << '\n';
        std::cout << std::flush;
    }
}

void finishLivePanel() {
    std::lock_guard<std::mutex> lock(gPrintMutex);
    if (gLiveRows > 0) {
        if (gLiveRows > 1) std::cout << "\033[" << gLiveRows - 1 << 'B';
        std::cout << '\n' << std::flush;
        gLiveRows = 0;
    }
}

struct Config {
    std::string worker;
    std::string engine = "resources/ATST-F_strong_1.trt";
    std::string mapping = "class_mapping.csv";
    std::string labels = "resources/ATST-F_strong_1.labels.txt";
    std::string videos = "videos";
    std::string ffmpeg = "ffmpeg";
    double mediaSeconds = 20.0;
    double reportSeconds = 5.0;
    double thresholdPercent = 10.0;
};

Config parseArguments(int argc, char** argv) {
    Config config;
    config.worker = (fs::path(argv[0]).parent_path() / "atst_sed_worker").string();
    for (int index = 1; index < argc; index += 2) {
        const std::string key = argv[index];
        if (key == "--help" || key == "-h") {
            std::cout << "Usage: " << argv[0] << " [--worker PATH] [--engine PATH]"
                      << " [--mapping PATH] [--labels PATH] [--videos-root PATH]"
                      << " [--ffmpeg PATH] [--media-seconds N] [--report-seconds N]"
                      << " [--threshold-percent N]\n"
                      << "Streams the videos playlist until Ctrl+C or SIGTERM.\n";
            std::exit(0);
        }
        if (index + 1 >= argc) throw std::runtime_error("Missing value after " + key);
        const std::string value = argv[index + 1];
        if (key == "--worker") config.worker = value;
        else if (key == "--engine") config.engine = value;
        else if (key == "--mapping") config.mapping = value;
        else if (key == "--labels") config.labels = value;
        else if (key == "--videos-root") config.videos = value;
        else if (key == "--ffmpeg") config.ffmpeg = value;
        else if (key == "--media-seconds") config.mediaSeconds = std::stod(value);
        else if (key == "--report-seconds") config.reportSeconds = std::stod(value);
        else if (key == "--threshold-percent") config.thresholdPercent = std::stod(value);
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (!std::isfinite(config.mediaSeconds) || config.mediaSeconds <= 0 ||
        !std::isfinite(config.reportSeconds) || config.reportSeconds <= 0) {
        throw std::runtime_error("Media and report intervals must be positive finite numbers");
    }
    if (!std::isfinite(config.thresholdPercent) || config.thresholdPercent < 0 ||
        config.thresholdPercent > 100) {
        throw std::runtime_error("Threshold percent must be between 0 and 100");
    }
    return config;
}

std::vector<fs::path> mediaFiles(const fs::path& root) {
    if (!fs::is_directory(root)) throw std::runtime_error("Media directory missing: " + root.string());
    const std::vector<std::string> extensions{
        ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v",
        ".wav", ".flac", ".mp3", ".ogg", ".oga", ".m4a", ".aac",
    };
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_symlink() || !entry.is_regular_file()) continue;
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        std::string left = a.filename().string();
        std::string right = b.filename().string();
        std::transform(left.begin(), left.end(), left.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::transform(right.begin(), right.end(), right.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return left < right;
    });
    if (files.empty()) throw std::runtime_error("No supported media files in " + root.string());
    return files;
}

struct Child {
    pid_t pid = -1;
    int input = -1;
    int output = -1;
};

Child spawn(const std::vector<std::string>& command, bool pipeInput, bool pipeOutput) {
    int input[2] = {-1, -1};
    int output[2] = {-1, -1};
    if ((pipeInput && ::pipe(input) != 0) || (pipeOutput && ::pipe(output) != 0)) {
        if (input[0] >= 0) { ::close(input[0]); ::close(input[1]); }
        throw std::runtime_error("pipe failed: " + std::string(std::strerror(errno)));
    }
    std::vector<char*> arguments;
    for (const auto& part : command) arguments.push_back(const_cast<char*>(part.c_str()));
    arguments.push_back(nullptr);
    const pid_t pid = ::fork();
    if (pid < 0) {
        if (input[0] >= 0) { ::close(input[0]); ::close(input[1]); }
        if (output[0] >= 0) { ::close(output[0]); ::close(output[1]); }
        throw std::runtime_error("fork failed: " + std::string(std::strerror(errno)));
    }
    if (pid == 0) {
        ::setpgid(0, 0);
        if (pipeInput) ::dup2(input[0], STDIN_FILENO);
        else {
            const int nullFd = ::open("/dev/null", O_RDONLY);
            if (nullFd >= 0) { ::dup2(nullFd, STDIN_FILENO); ::close(nullFd); }
        }
        if (pipeOutput) ::dup2(output[1], STDOUT_FILENO);
        if (input[0] >= 0) { ::close(input[0]); ::close(input[1]); }
        if (output[0] >= 0) { ::close(output[0]); ::close(output[1]); }
        ::execvp(arguments[0], arguments.data());
        _exit(127);
    }
    if (pipeInput) ::close(input[0]);
    if (pipeOutput) ::close(output[1]);
    return {pid, pipeInput ? input[1] : -1, pipeOutput ? output[0] : -1};
}

std::optional<int> waitForExit(pid_t pid, std::chrono::milliseconds timeout) {
    const auto deadline = Clock::now() + timeout;
    int status = 0;
    do {
        const pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        if (result < 0 && errno != EINTR) return std::nullopt;
        std::this_thread::sleep_for(25ms);
    } while (Clock::now() < deadline);
    return std::nullopt;
}

int finishChild(Child& child, bool stop, std::chrono::milliseconds timeout = 5s) {
    if (child.input >= 0) { ::close(child.input); child.input = -1; }
    if (child.output >= 0) { ::close(child.output); child.output = -1; }
    if (child.pid < 0) return 0;
    if (stop) ::kill(child.pid, SIGTERM);
    auto exitCode = waitForExit(child.pid, timeout);
    if (!exitCode) {
        ::kill(child.pid, SIGKILL);
        exitCode = waitForExit(child.pid, 1s);
    }
    child.pid = -1;
    return exitCode.value_or(-1);
}

void writeMessage(int fd, const json& message) {
    const std::string line = message.dump() + '\n';
    std::size_t offset = 0;
    while (offset < line.size()) {
        pollfd descriptor{fd, POLLOUT, 0};
        const int ready = ::poll(&descriptor, 1, 2000);
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL))) {
            throw std::runtime_error("Worker stopped accepting messages");
        }
        const ssize_t count = ::write(fd, line.data() + offset, line.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Worker input pipe closed");
        offset += static_cast<std::size_t>(count);
    }
}

std::string base64(const char* data, std::size_t size) {
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((size + 2) / 3 * 4);
    for (std::size_t offset = 0; offset < size; offset += 3) {
        const unsigned a = static_cast<unsigned char>(data[offset]);
        const unsigned b = offset + 1 < size ? static_cast<unsigned char>(data[offset + 1]) : 0;
        const unsigned c = offset + 2 < size ? static_cast<unsigned char>(data[offset + 2]) : 0;
        encoded.push_back(alphabet[a >> 2]);
        encoded.push_back(alphabet[((a & 3) << 4) | (b >> 4)]);
        encoded.push_back(offset + 1 < size ? alphabet[((b & 15) << 2) | (c >> 6)] : '=');
        encoded.push_back(offset + 2 < size ? alphabet[c & 63] : '=');
    }
    return encoded;
}

std::string displayClassName(const std::string& name) {
    std::string shortName = name.substr(0, name.find('('));
    while (!shortName.empty() && std::isspace(static_cast<unsigned char>(shortName.back()))) {
        shortName.pop_back();
    }
    return shortName;
}

struct Telemetry {
    std::mutex mutex;
    std::condition_variable changed;
    Clock::time_point launched = Clock::now();
    Clock::time_point streamStarted = Clock::now();
    Clock::time_point lastResult = Clock::now();
    std::string media = "waiting for worker";
    std::size_t mediaIndex = 0;
    std::size_t mediaCount = 0;
    std::uint64_t completedMedia = 0;
    double segmentSeconds = 20.0;
    std::string error;
    std::vector<std::string> classes;
    std::vector<double> latestScores;
    double thresholdPercent = 10.0;
    bool ready = false;
    bool eof = false;
    bool complete = false;
    std::int64_t streamId = 0;
    std::uint64_t packets = 0;
    std::uint64_t streamPackets = 0;
    std::uint64_t results = 0;
    std::uint64_t streamResults = 0;
    std::uint64_t supersededTotal = 0;
    int inferenceMs = -1;
    int lagMs = -1;
    int superseded = -1;

    void onCallback(const json& event) {
        const std::string kind = event.value("event", "");
        bool showResult = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (kind == "ready") {
                classes = event.at("classes").get<std::vector<std::string>>();
                if (classes.empty()) error = "Worker returned no classes";
                else ready = true;
            } else if (kind == "result") {
                ++results;
                inferenceMs = event.value("processing_ms", -1);
                superseded = event.value("superseded_packets", 0);
                supersededTotal += static_cast<std::uint64_t>(superseded);
                if (event.value("id", std::int64_t(-1)) == streamId) {
                    ++streamResults;
                    lastResult = Clock::now();
                    const auto timestampMs = event.value("timestamp_ms", std::int64_t(0));
                    lagMs = std::max(0, static_cast<int>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(lastResult - streamStarted).count()
                        - timestampMs));
                    const auto scores = event.at("scores").get<std::vector<double>>();
                    if (scores.size() != classes.size()) {
                        error = "Worker score count differs from class count";
                    } else {
                        latestScores = scores;
                        showResult = true;
                    }
                }
            } else if (kind == "complete" && event.value("id", std::int64_t(-1)) == streamId) {
                complete = true;
            } else if (kind == "error" || kind == "fatal") {
                error = event.value("message", "Worker error");
            }
        }
        changed.notify_all();
        if (showResult && ::isatty(STDOUT_FILENO)) report();
    }

    void check() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error.empty()) throw std::runtime_error(error);
        if (eof) throw std::runtime_error("Worker exited unexpectedly");
        if (streamId != 0 && packets > 0 && Clock::now() - lastResult > 10s) {
            throw std::runtime_error("Worker returned no result for 10 seconds");
        }
    }

    void report() {
        std::vector<std::string> lines;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!ready) return;
            const auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
                Clock::now() - launched).count();
            std::ostringstream header;
            header << "Uptime: " << uptime << " s | Inference: " << inferenceMs
                   << " ms | Lag: " << lagMs << " ms | Superseded: " << superseded;
            lines.push_back(header.str());
            std::ostringstream mediaLine;
            mediaLine << "Media " << mediaIndex << '/' << mediaCount
                      << " | completed: " << completedMedia << " | " << media;
            lines.push_back(mediaLine.str());
            constexpr std::size_t barWidth = 24;
            const double playedSeconds = streamPackets * kPacketMs / 1000.0;
            const double progress = std::clamp(playedSeconds / segmentSeconds, 0.0, 1.0);
            const auto filled = static_cast<std::size_t>(std::round(progress * barWidth));
            std::ostringstream playback;
            playback << "Playback segment: [" << std::string(filled, '#')
                     << std::string(barWidth - filled, '-') << "] "
                     << std::fixed << std::setprecision(1) << playedSeconds
                     << '/' << segmentSeconds << " s (" << std::setprecision(0)
                     << progress * 100 << "%)";
            lines.push_back(playback.str());
            std::ostringstream triggered;
            triggered << std::fixed << std::setprecision(1);
            bool anyTriggered = false;
            for (std::size_t index = 0; index < classes.size(); ++index) {
                std::ostringstream row;
                row << std::left << std::setw(24) << displayClassName(classes[index]);
                if (index < latestScores.size()) {
                    const double confidence = latestScores[index] * 100.0;
                    row << std::right << std::fixed << std::setprecision(1)
                        << std::setw(6) << confidence << '%';
                    if (confidence > thresholdPercent) {
                        row << "  TRIGGERED";
                        if (anyTriggered) triggered << ", ";
                        triggered << displayClassName(classes[index]);
                        anyTriggered = true;
                    }
                } else {
                    row << "waiting";
                }
                lines.push_back(row.str());
            }
            std::ostringstream footer;
            footer << std::fixed << std::setprecision(1)
                   << "Triggered (above " << thresholdPercent << "%): "
                   << (anyTriggered ? triggered.str() : "none");
            lines.push_back(footer.str());
        }
        showLivePanel(lines);
    }
};

void readCallbacks(int fd, Telemetry& telemetry) {
    FILE* stream = ::fdopen(fd, "r");
    if (!stream) {
        std::lock_guard<std::mutex> lock(telemetry.mutex);
        telemetry.error = "Could not read worker callbacks";
        telemetry.changed.notify_all();
        ::close(fd);
        return;
    }
    char* line = nullptr;
    std::size_t capacity = 0;
    while (::getline(&line, &capacity, stream) >= 0) {
        try { telemetry.onCallback(json::parse(line)); }
        catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(telemetry.mutex);
            telemetry.error = std::string("Invalid worker callback: ") + error.what();
            telemetry.changed.notify_all();
        }
    }
    std::free(line);
    ::fclose(stream);
    std::lock_guard<std::mutex> lock(telemetry.mutex);
    telemetry.eof = true;
    telemetry.changed.notify_all();
}

void streamMedia(const Config& config, const fs::path& path, std::int64_t id,
                 std::size_t mediaIndex,
                 Child& worker, Telemetry& telemetry, Child& decoder) {
    {
        std::lock_guard<std::mutex> lock(telemetry.mutex);
        telemetry.streamId = id;
        telemetry.media = path.filename().string();
        telemetry.mediaIndex = mediaIndex;
        telemetry.segmentSeconds = config.mediaSeconds;
        telemetry.streamPackets = 0;
        telemetry.streamResults = 0;
        telemetry.complete = false;
        telemetry.lastResult = Clock::now();
        telemetry.latestScores.clear();
    }
    telemetry.report();
    writeMessage(worker.input, {{"type", "stream_start"}, {"id", id},
                                {"cam_id", "long-stream-test"}, {"timestamp_ms", 0}});
    decoder = spawn({config.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                     "-i", path.string(), "-vn", "-map", "0:a:0", "-ac", "1",
                     "-ar", "16000", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"},
                    false, true);
    const auto started = Clock::now();
    const auto segmentEnd = started + std::chrono::duration<double>(config.mediaSeconds);
    std::string buffer;
    std::uint64_t packetIndex = 0;
    bool eof = false;
    while (!gStop && Clock::now() < segmentEnd) {
        telemetry.check();
        if (buffer.size() < kPacketBytes) {
            pollfd descriptor{decoder.output, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, 100);
            if (ready < 0 && errno == EINTR) continue;
            if (ready < 0) throw std::runtime_error("FFmpeg poll failed");
            if (ready > 0 && (descriptor.revents & (POLLIN | POLLHUP))) {
                char chunk[4096];
                const ssize_t count = ::read(decoder.output, chunk, sizeof(chunk));
                if (count < 0 && errno == EINTR) continue;
                if (count < 0) throw std::runtime_error("FFmpeg output read failed");
                if (count == 0) eof = true;
                else buffer.append(chunk, static_cast<std::size_t>(count));
            }
            if (buffer.size() < kPacketBytes && !eof) continue;
            if (buffer.empty() && eof) break;
        }
        std::string packet = buffer.substr(0, kPacketBytes);
        buffer.erase(0, packet.size());
        packet.resize(kPacketBytes, '\0');
        const auto deadline = started + std::chrono::milliseconds(packetIndex * kPacketMs);
        while (!gStop && Clock::now() < deadline) {
            std::this_thread::sleep_for(std::min(20ms,
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now())));
        }
        if (gStop) break;
        if (packetIndex == 0) {
            std::lock_guard<std::mutex> lock(telemetry.mutex);
            telemetry.streamStarted = Clock::now();
            telemetry.lastResult = telemetry.streamStarted;
        }
        writeMessage(worker.input, {{"type", "audio"}, {"id", id},
                                    {"cam_id", "long-stream-test"},
                                    {"timestamp_ms", packetIndex * kPacketMs},
                                    {"sample_rate", 16000}, {"channels", 1},
                                    {"encoding", "s16le"},
                                    {"audio_b64", base64(packet.data(), packet.size())}});
        ++packetIndex;
        {
            std::lock_guard<std::mutex> lock(telemetry.mutex);
            ++telemetry.packets;
            ++telemetry.streamPackets;
        }
        if (eof && buffer.empty()) break;
    }
    const int decoderExit = finishChild(decoder, !eof || gStop, 1s);
    if (!gStop && eof && decoderExit != 0) {
        throw std::runtime_error("FFmpeg failed for " + path.string() + " (exit " +
                                 std::to_string(decoderExit) + ")");
    }
    if (gStop) {
        // Close the active stream before closing the worker's input pipe.
        try {
            writeMessage(worker.input, {{"type", "stream_end"}, {"id", id},
                                        {"cam_id", "long-stream-test"},
                                        {"timestamp_ms", packetIndex * kPacketMs}});
        } catch (const std::exception&) {
            // Shutdown will still close the pipe if the worker has already exited.
        }
        return;
    }
    if (packetIndex == 0) throw std::runtime_error("No audio packets from " + path.string());
    writeMessage(worker.input, {{"type", "stream_end"}, {"id", id},
                                {"cam_id", "long-stream-test"},
                                {"timestamp_ms", packetIndex * kPacketMs}});
    const auto timeout = Clock::now() + 10s;
    std::unique_lock<std::mutex> lock(telemetry.mutex);
    while (!gStop && !telemetry.complete && telemetry.error.empty() && !telemetry.eof &&
           Clock::now() < timeout) {
        telemetry.changed.wait_for(lock, 100ms);
    }
    if (gStop) return;
    if (!telemetry.error.empty()) throw std::runtime_error(telemetry.error);
    if (!telemetry.complete) throw std::runtime_error("Worker did not acknowledge stream_end");
    if (telemetry.streamResults == 0) throw std::runtime_error("Worker produced no result for " + path.string());
    ++telemetry.completedMedia;
}

int run(const Config& config) {
    const auto files = mediaFiles(config.videos);
    Child worker = spawn({config.worker, config.engine, config.mapping, config.labels}, true, true);
    Telemetry telemetry;
    telemetry.thresholdPercent = config.thresholdPercent;
    telemetry.mediaCount = files.size();
    const int callbackFd = worker.output;
    worker.output = -1;  // The reader thread owns this descriptor.
    std::thread reader([&] { readCallbacks(callbackFd, telemetry); });
    std::atomic<bool> stopReporter{false};
    std::thread reporter([&] {
        auto next = Clock::now() + std::chrono::duration<double>(config.reportSeconds);
        while (!stopReporter && !gStop) {
            std::this_thread::sleep_for(100ms);
            if (Clock::now() >= next) {
                telemetry.report();
                next = Clock::now() + std::chrono::duration<double>(config.reportSeconds);
            }
        }
    });
    Child decoder;
    std::string failure;
    try {
        const auto deadline = Clock::now() + 120s;
        std::unique_lock<std::mutex> lock(telemetry.mutex);
        while (!gStop && !telemetry.ready && telemetry.error.empty() && !telemetry.eof &&
               Clock::now() < deadline) {
            telemetry.changed.wait_for(lock, 100ms);
        }
        if (!gStop && !telemetry.error.empty()) throw std::runtime_error(telemetry.error);
        if (!gStop && !telemetry.ready) throw std::runtime_error("Worker did not become ready");
        lock.unlock();
        std::int64_t streamId = 0;
        while (!gStop) {
            for (std::size_t index = 0; index < files.size(); ++index) {
                if (gStop) break;
                streamMedia(config, files[index], ++streamId, index + 1,
                            worker, telemetry, decoder);
            }
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }
    if (decoder.pid > 0) finishChild(decoder, true, 1s);
    stopReporter = true;
    reporter.join();
    telemetry.report();
    const int workerExit = finishChild(worker, !failure.empty(), 10s);
    reader.join();
    if (failure.empty() && workerExit != 0) failure = "Worker failed during shutdown";
    if (!failure.empty()) {
        finishLivePanel();
        std::cerr << "FAIL: " << failure << '\n';
        return 1;
    }
    log("Stopped by user; worker exited cleanly");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
    std::signal(SIGPIPE, SIG_IGN);
    try { return run(parseArguments(argc, argv)); }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
