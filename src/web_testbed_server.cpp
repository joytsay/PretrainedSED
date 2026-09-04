#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr std::size_t kMaxHeaderBytes = 64 * 1024;
constexpr std::size_t kMaxBodyBytes = 4 * 1024 * 1024;
constexpr std::size_t kMaxEvents = 4096;
constexpr auto kWorkerIdleTimeout = std::chrono::minutes(5);

std::int64_t steadyNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::atomic<bool> gRunning{true};
std::atomic<int> gListenFd{-1};

void handleSignal(int) {
    gRunning = false;
    const int fd = gListenFd.exchange(-1);
    if (fd >= 0) ::close(fd);
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<std::string> parseCsvRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char c = line[index];
        if (quoted) {
            if (c == '"' && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else if (c == '"') {
                quoted = false;
            } else {
                field.push_back(c);
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    if (quoted) throw std::runtime_error("Unclosed quote in class mapping CSV");
    fields.push_back(trim(field));
    return fields;
}

std::string readTextFile(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void validateMapping(const std::string& csv, const fs::path& labelsPath) {
    std::unordered_set<std::string> labels;
    std::istringstream labelInput(readTextFile(labelsPath));
    std::string line;
    while (std::getline(labelInput, line)) {
        line = trim(line);
        if (!line.empty()) labels.insert(lower(line));
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> groups;
    std::istringstream input(csv);
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (trim(line).empty() || trim(line).front() == '#') continue;
        const auto fields = parseCsvRow(line);
        if (fields.size() >= 2 && lower(fields[0]) == "class_name" &&
            lower(fields[1]) == "source_class") {
            continue;
        }
        if (fields.size() < 2 || fields[0].empty() || fields[1].empty()) {
            throw std::runtime_error("Mapping line " + std::to_string(lineNumber) +
                                     " needs class_name,source_class");
        }
        const std::string sourceKey = lower(fields[1]);
        if (labels.find(sourceKey) == labels.end()) {
            throw std::runtime_error("Mapping line " + std::to_string(lineNumber) +
                                     " has an unknown source class: " + fields[1]);
        }
        auto& sources = groups[fields[0]];
        if (!sources.insert(sourceKey).second) {
            throw std::runtime_error("Mapping line " + std::to_string(lineNumber) +
                                     " duplicates source class " + fields[1]);
        }
    }
    if (groups.empty()) throw std::runtime_error("Mapping must define at least one aggregate class");
}

class EventLog {
public:
    std::uint64_t push(json value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (events_.size() >= kMaxEvents) {
            // A new browser must not be stranded behind an old sequence.  A
            // zero sequence tells clients to restart their event cursor.
            events_.clear();
            sequence_ = 0;
        }
        const std::uint64_t sequence = ++sequence_;
        events_.push_back({sequence, std::move(value)});
        while (events_.size() > kMaxEvents) events_.pop_front();
        condition_.notify_all();
        return sequence;
    }

    json after(std::uint64_t sequence, std::chrono::milliseconds wait) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, wait, [&] { return sequence_ > sequence || !gRunning.load(); });
        json output = json::array();
        for (const auto& event : events_) {
            if (event.first > sequence) {
                output.push_back({{"seq", event.first}, {"data", event.second}});
            }
        }
        return {{"events", output}, {"next", sequence_}};
    }

    void wakeAll() { condition_.notify_all(); }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::pair<std::uint64_t, json>> events_;
    std::uint64_t sequence_ = 0;
};

struct WorkerConfig {
    std::string executable;
    std::string engine;
    std::string mapping;
    std::string labels;
};

class WorkerProcess {
public:
    WorkerProcess(WorkerConfig config, EventLog& events)
        : config_(std::move(config)), events_(events) {}

    ~WorkerProcess() { stop(); }

    void start() {
        std::lock_guard<std::mutex> lifecycle(lifecycleMutex_);
        startLocked();
    }

    void restart() {
        std::lock_guard<std::mutex> lifecycle(lifecycleMutex_);
        stopLocked();
        events_.push({{"event", "worker_restarting"}});
        startLocked();
    }

    void stop() {
        std::lock_guard<std::mutex> lifecycle(lifecycleMutex_);
        stopLocked();
    }

    void send(const json& message) {
        const std::string line = message.dump() + "\n";
        std::lock_guard<std::mutex> lock(writeMutex_);
        if (inputFd_ < 0) throw std::runtime_error("Callback worker is not running");
        std::size_t offset = 0;
        while (offset < line.size()) {
            const ssize_t count = ::write(inputFd_, line.data() + offset, line.size() - offset);
            if (count < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("Could not write to callback worker: " +
                                         std::string(std::strerror(errno)));
            }
            offset += static_cast<std::size_t>(count);
        }
    }

    bool running() const { return pid_.load() > 0; }
    bool ready() const { return ready_.load(); }

private:
    void startLocked() {
        if (pid_.load() > 0) return;
        int inputPipe[2] = {-1, -1};
        int outputPipe[2] = {-1, -1};
        int errorPipe[2] = {-1, -1};
        if (::pipe(inputPipe) || ::pipe(outputPipe) || ::pipe(errorPipe)) {
            throw std::runtime_error("pipe() failed: " + std::string(std::strerror(errno)));
        }
        const pid_t child = ::fork();
        if (child < 0) throw std::runtime_error("fork() failed");
        if (child == 0) {
            ::dup2(inputPipe[0], STDIN_FILENO);
            ::dup2(outputPipe[1], STDOUT_FILENO);
            ::dup2(errorPipe[1], STDERR_FILENO);
            ::close(inputPipe[0]); ::close(inputPipe[1]);
            ::close(outputPipe[0]); ::close(outputPipe[1]);
            ::close(errorPipe[0]); ::close(errorPipe[1]);
            ::execl(config_.executable.c_str(), config_.executable.c_str(),
                    config_.engine.c_str(), config_.mapping.c_str(), config_.labels.c_str(),
                    static_cast<char*>(nullptr));
            std::fprintf(stderr, "exec failed: %s\n", std::strerror(errno));
            _exit(127);
        }
        ::close(inputPipe[0]);
        ::close(outputPipe[1]);
        ::close(errorPipe[1]);
        inputFd_ = inputPipe[1];
        outputFd_ = outputPipe[0];
        errorFd_ = errorPipe[0];
        pid_ = child;
        ready_ = false;
        outputThread_ = std::thread([this] { readOutput(); });
        errorThread_ = std::thread([this] { readErrors(); });
    }

    void stopLocked() {
        const pid_t child = pid_.exchange(-1);
        ready_ = false;
        {
            std::lock_guard<std::mutex> lock(writeMutex_);
            if (inputFd_ >= 0) ::close(std::exchange(inputFd_, -1));
        }
        if (child > 0) {
            int status = 0;
            for (int attempt = 0; attempt < 20; ++attempt) {
                if (::waitpid(child, &status, WNOHANG) == child) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            if (::waitpid(child, &status, WNOHANG) == 0) {
                ::kill(child, SIGTERM);
                ::waitpid(child, &status, 0);
            }
        }
        if (outputThread_.joinable()) outputThread_.join();
        if (errorThread_.joinable()) errorThread_.join();
        // fdopen/fclose in the reader threads owns and closes these descriptors.
        outputFd_ = -1;
        errorFd_ = -1;
    }

    void readOutput() {
        FILE* stream = ::fdopen(outputFd_, "r");
        if (!stream) return;
        char* line = nullptr;
        std::size_t capacity = 0;
        while (::getline(&line, &capacity, stream) >= 0) {
            try {
                json callback = json::parse(line);
                if (callback.value("event", "") == "ready") ready_ = true;
                if (callback.value("event", "") == "fatal") ready_ = false;
                events_.push(std::move(callback));
            } catch (const std::exception& error) {
                events_.push({{"event", "server_error"},
                              {"message", std::string("Invalid worker callback: ") + error.what()}});
            }
        }
        std::free(line);
        // fdopen owns outputFd_; leave the numeric member for stopLocked to reset.
        ::fclose(stream);
        ready_ = false;
    }

    void readErrors() {
        FILE* stream = ::fdopen(errorFd_, "r");
        if (!stream) return;
        char* line = nullptr;
        std::size_t capacity = 0;
        while (::getline(&line, &capacity, stream) >= 0) {
            const std::string message = trim(line);
            if (!message.empty()) {
                events_.push({{"event", "worker_log"}, {"message", message}});
            }
        }
        std::free(line);
        ::fclose(stream);
    }

    WorkerConfig config_;
    EventLog& events_;
    std::atomic<pid_t> pid_{-1};
    std::atomic<bool> ready_{false};
    int inputFd_ = -1;
    int outputFd_ = -1;
    int errorFd_ = -1;
    std::thread outputThread_;
    std::thread errorThread_;
    std::mutex lifecycleMutex_;
    std::mutex writeMutex_;
};

struct HttpRequest {
    std::string method;
    std::string target;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

void sendAll(int fd, const std::string& value) {
    std::size_t offset = 0;
    while (offset < value.size()) {
        const ssize_t count = ::send(fd, value.data() + offset, value.size() - offset, MSG_NOSIGNAL);
        if (count < 0) {
            if (errno == EINTR) continue;
            return;
        }
        offset += static_cast<std::size_t>(count);
    }
}

void respond(int fd, int status, const std::string& contentType, const std::string& body,
             const std::string& extraHeaders = {}) {
    const char* reason = status == 200 ? "OK" : status == 206 ? "Partial Content" :
                         status == 204 ? "No Content" :
                         status == 400 ? "Bad Request" : status == 404 ? "Not Found" :
                         status == 405 ? "Method Not Allowed" :
                         status == 416 ? "Range Not Satisfiable" : "Internal Server Error";
    std::ostringstream output;
    output << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
           << "Content-Type: " << contentType << "\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Cache-Control: no-store\r\n"
           << "Connection: close\r\n" << extraHeaders << "\r\n" << body;
    sendAll(fd, output.str());
}

std::optional<HttpRequest> readRequest(int fd) {
    std::string data;
    char buffer[8192];
    std::size_t headerEnd = std::string::npos;
    while ((headerEnd = data.find("\r\n\r\n")) == std::string::npos) {
        const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) return std::nullopt;
        data.append(buffer, static_cast<std::size_t>(count));
        if (data.size() > kMaxHeaderBytes) throw std::runtime_error("HTTP header is too large");
    }
    HttpRequest request;
    std::istringstream headers(data.substr(0, headerEnd));
    std::string version;
    headers >> request.method >> request.target >> version;
    std::string line;
    std::getline(headers, line);
    while (std::getline(headers, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        request.headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }
    std::size_t contentLength = 0;
    if (const auto found = request.headers.find("content-length"); found != request.headers.end()) {
        contentLength = static_cast<std::size_t>(std::stoull(found->second));
    }
    if (contentLength > kMaxBodyBytes) throw std::runtime_error("HTTP body is too large");
    request.body = data.substr(headerEnd + 4);
    while (request.body.size() < contentLength) {
        const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) throw std::runtime_error("HTTP body ended early");
        request.body.append(buffer, static_cast<std::size_t>(count));
    }
    request.body.resize(contentLength);
    return request;
}

std::string pathOnly(const std::string& target) {
    const auto query = target.find('?');
    return target.substr(0, query);
}

std::string queryValue(const std::string& target, const std::string& key) {
    const auto query = target.find('?');
    if (query == std::string::npos) return {};
    std::istringstream pairs(target.substr(query + 1));
    std::string pair;
    while (std::getline(pairs, pair, '&')) {
        const auto equal = pair.find('=');
        if (pair.substr(0, equal) == key) {
            return equal == std::string::npos ? std::string() : pair.substr(equal + 1);
        }
    }
    return {};
}

std::string percentEncode(const std::string& value) {
    constexpr char kHex[] = "0123456789ABCDEF";
    std::string encoded;
    for (const unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            encoded.push_back(static_cast<char>(c));
        } else {
            encoded.push_back('%');
            encoded.push_back(kHex[c >> 4]);
            encoded.push_back(kHex[c & 0x0f]);
        }
    }
    return encoded;
}

int hexDigit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

std::string percentDecode(const std::string& value) {
    std::string decoded;
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] != '%') {
            decoded.push_back(value[index]);
            continue;
        }
        if (index + 2 >= value.size()) throw std::runtime_error("Invalid media URL encoding");
        const int high = hexDigit(value[index + 1]);
        const int low = hexDigit(value[index + 2]);
        if (high < 0 || low < 0) throw std::runtime_error("Invalid media URL encoding");
        decoded.push_back(static_cast<char>((high << 4) | low));
        index += 2;
    }
    return decoded;
}

std::string mimeType(const fs::path& path) {
    const std::string extension = lower(path.extension().string());
    if (extension == ".html") return "text/html; charset=utf-8";
    if (extension == ".js") return "text/javascript; charset=utf-8";
    if (extension == ".css") return "text/css; charset=utf-8";
    if (extension == ".svg") return "image/svg+xml";
    if (extension == ".png") return "image/png";
    if (extension == ".ico") return "image/x-icon";
    if (extension == ".mp4" || extension == ".m4v") return "video/mp4";
    if (extension == ".webm") return "video/webm";
    if (extension == ".mov") return "video/quicktime";
    if (extension == ".mkv") return "video/x-matroska";
    if (extension == ".avi") return "video/x-msvideo";
    if (extension == ".mp3") return "audio/mpeg";
    if (extension == ".wav") return "audio/wav";
    if (extension == ".flac") return "audio/flac";
    if (extension == ".ogg" || extension == ".oga") return "audio/ogg";
    if (extension == ".m4a") return "audio/mp4";
    if (extension == ".aac") return "audio/aac";
    return "application/octet-stream";
}

bool isSupportedMedia(const fs::path& path) {
    static const std::unordered_set<std::string> extensions{
        ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v",
        ".wav", ".flac", ".mp3", ".ogg", ".oga", ".m4a", ".aac",
    };
    return extensions.find(lower(path.extension().string())) != extensions.end();
}

void respondFile(int fd, const fs::path& path, const std::string& rangeHeader) {
    const std::uintmax_t size = fs::file_size(path);
    std::uintmax_t start = 0;
    std::uintmax_t end = size == 0 ? 0 : size - 1;
    int status = 200;

    if (!rangeHeader.empty() && size > 0) {
        try {
            if (rangeHeader.rfind("bytes=", 0) != 0 || rangeHeader.find(',') != std::string::npos) {
                throw std::runtime_error("unsupported range");
            }
            const std::string range = rangeHeader.substr(6);
            const auto dash = range.find('-');
            if (dash == std::string::npos) throw std::runtime_error("invalid range");
            const std::string first = range.substr(0, dash);
            const std::string last = range.substr(dash + 1);
            if (first.empty()) {
                const std::uintmax_t suffix = std::stoull(last);
                if (suffix == 0) throw std::runtime_error("invalid suffix range");
                start = suffix >= size ? 0 : size - suffix;
            } else {
                start = std::stoull(first);
                if (!last.empty()) end = std::min<std::uintmax_t>(std::stoull(last), size - 1);
            }
            if (start >= size || start > end) throw std::runtime_error("range outside file");
            status = 206;
        } catch (const std::exception&) {
            respond(fd, 416, "text/plain", {},
                    "Content-Range: bytes */" + std::to_string(size) + "\r\n");
            return;
        }
    }

    const std::uintmax_t contentLength = size == 0 ? 0 : end - start + 1;
    std::ostringstream header;
    header << "HTTP/1.1 " << status << (status == 206 ? " Partial Content" : " OK") << "\r\n"
           << "Content-Type: " << mimeType(path) << "\r\n"
           << "Content-Length: " << contentLength << "\r\n"
           << "Accept-Ranges: bytes\r\n";
    if (status == 206) {
        header << "Content-Range: bytes " << start << '-' << end << '/' << size << "\r\n";
    }
    header << "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    sendAll(fd, header.str());

    if (contentLength == 0) return;
    std::ifstream input(path, std::ios::binary);
    if (!input) return;
    input.seekg(static_cast<std::streamoff>(start));
    std::uintmax_t remaining = contentLength;
    std::string buffer(64 * 1024, '\0');
    while (remaining > 0 && input) {
        const std::size_t wanted = static_cast<std::size_t>(
            std::min<std::uintmax_t>(remaining, buffer.size()));
        input.read(buffer.data(), static_cast<std::streamsize>(wanted));
        const std::streamsize count = input.gcount();
        if (count <= 0) break;
        sendAll(fd, std::string(buffer.data(), static_cast<std::size_t>(count)));
        remaining -= static_cast<std::uintmax_t>(count);
    }
}

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    fs::path webRoot = "src/webui/dist";
    fs::path videosRoot = "/workspace/videos";
    WorkerConfig worker;
};

class WebTestbedServer {
public:
    explicit WebTestbedServer(ServerConfig config)
        : config_(std::move(config)), worker_(config_.worker, events_) {}

    void run() {
        worker_.start();
        const int server = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server < 0) throw std::runtime_error("socket() failed");
        int enabled = 1;
        ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<std::uint16_t>(config_.port));
        if (::inet_pton(AF_INET, config_.host.c_str(), &address.sin_addr) != 1) {
            ::close(server);
            throw std::runtime_error("--host must be an IPv4 address");
        }
        if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
            ::listen(server, 64) < 0) {
            const std::string message = std::strerror(errno);
            ::close(server);
            throw std::runtime_error("Cannot listen on " + config_.host + ":" +
                                     std::to_string(config_.port) + ": " + message);
        }
        gListenFd = server;
        lastWorkerUseMs_ = steadyNowMs();
        idleThread_ = std::thread([this] { monitorWorkerIdle(); });
        std::cout << "GeoVision SED web testbed listening on http://" << config_.host << ':'
                  << config_.port << '\n';
        while (gRunning) {
            const int client = ::accept(server, nullptr, nullptr);
            if (client < 0) {
                if (errno == EINTR) continue;
                if (!gRunning) break;
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(clientMutex_);
                ++activeClients_;
                clientFds_.insert(client);
            }
            std::thread([this, client] {
                try { handleClient(client); }
                catch (const std::exception& error) {
                    respond(client, 500, "application/json", json({{"error", error.what()}}).dump());
                }
                ::close(client);
                {
                    std::lock_guard<std::mutex> lock(clientMutex_);
                    clientFds_.erase(client);
                    --activeClients_;
                }
                clientCondition_.notify_all();
            }).detach();
        }
        gListenFd = -1;
        // SIGINT/SIGTERM makes the long-poll predicate true, but changing an
        // atomic does not wake a condition variable by itself. Wake all
        // /api/events requests so shutdown does not wait for their 20-second
        // timeout before joining the active client threads.
        events_.wakeAll();
        {
            std::lock_guard<std::mutex> lock(clientMutex_);
            // Firefox may keep media range requests open. Interrupt all active
            // reads and writes; each client thread remains responsible for
            // closing its own descriptor.
            for (const int client : clientFds_) ::shutdown(client, SHUT_RDWR);
        }
        std::unique_lock<std::mutex> clientLock(clientMutex_);
        clientCondition_.wait(clientLock, [this] { return activeClients_ == 0; });
        clientLock.unlock();
        idleStopping_ = true;
        if (idleThread_.joinable()) idleThread_.join();
        worker_.stop();
    }

private:
    void handleClient(int fd) {
        const auto requestValue = readRequest(fd);
        if (!requestValue) return;
        const HttpRequest& request = *requestValue;
        const std::string path = pathOnly(request.target);
        if (request.method == "OPTIONS") {
            respond(fd, 204, "text/plain", {});
        } else if (path == "/api/health" && request.method == "GET") {
            respond(fd, 200, "application/json",
                    json({{"ok", true}, {"worker_running", worker_.running()},
                          {"worker_ready", worker_.ready()}}).dump());
        } else if (path == "/api/events" && request.method == "GET") {
            std::uint64_t after = 0;
            const std::string value = queryValue(request.target, "after");
            if (!value.empty()) after = std::stoull(value);
            respond(fd, 200, "application/json",
                    events_.after(after, std::chrono::seconds(20)).dump());
        } else if (path == "/api/message" && request.method == "POST") {
            const json message = json::parse(request.body);
            if (!message.is_object()) throw std::runtime_error("Message must be a JSON object");
            lastWorkerUseMs_ = steadyNowMs();
            if (!worker_.running()) worker_.start();
            worker_.send(message);
            respond(fd, 200, "application/json", "{\"ok\":true}");
        } else if (path == "/api/mapping" && request.method == "GET") {
            respond(fd, 200, "text/csv; charset=utf-8", readTextFile(config_.worker.mapping));
        } else if (path == "/api/videos" && request.method == "GET") {
            respond(fd, 200, "application/json", listVideos().dump());
        } else if (path == "/api/mapping" && request.method == "PUT") {
            validateMapping(request.body, config_.worker.labels);
            const fs::path mappingPath = config_.worker.mapping;
            const fs::path temporary = mappingPath.string() + ".web-testbed.tmp";
            {
                std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
                if (!output) throw std::runtime_error("Cannot write " + temporary.string());
                output << request.body;
                if (!output) throw std::runtime_error("Could not finish writing mapping CSV");
            }
            fs::rename(temporary, mappingPath);
            worker_.restart();
            respond(fd, 200, "application/json", "{\"ok\":true,\"restarting_worker\":true}");
        } else if (path.rfind("/api/", 0) == 0) {
            respond(fd, 404, "application/json", "{\"error\":\"API route not found\"}");
        } else if (path.rfind("/media/", 0) == 0 && request.method == "GET") {
            serveMedia(fd, request, path.substr(7));
        } else if (request.method == "GET") {
            serveStatic(fd, path);
        } else {
            respond(fd, 405, "application/json", "{\"error\":\"Method not allowed\"}");
        }
    }

    void monitorWorkerIdle() {
        while (!idleStopping_) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            if (idleStopping_) break;
            if (worker_.running() && steadyNowMs() - lastWorkerUseMs_.load() >
                                      std::chrono::duration_cast<std::chrono::milliseconds>(
                                          kWorkerIdleTimeout).count()) {
                worker_.stop();
                events_.push({{"event", "worker_sleeping"},
                              {"message", "Worker stopped after idle timeout"}});
            }
        }
    }

    json listVideos() const {
        json videos = json::array();
        std::error_code error;
        if (!fs::is_directory(config_.videosRoot, error)) {
            return {{"root", config_.videosRoot.string()}, {"videos", videos}};
        }
        std::vector<fs::directory_entry> entries;
        for (const auto& entry : fs::directory_iterator(config_.videosRoot, error)) {
            if (error) break;
            if (entry.is_symlink(error) || !entry.is_regular_file(error) ||
                !isSupportedMedia(entry.path())) {
                continue;
            }
            entries.push_back(entry);
        }
        std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
            return lower(left.path().filename().string()) < lower(right.path().filename().string());
        });
        for (const auto& entry : entries) {
            const std::string name = entry.path().filename().string();
            videos.push_back({{"name", name},
                              {"size", entry.file_size(error)},
                              {"url", "/media/" + percentEncode(name)}});
        }
        return {{"root", config_.videosRoot.string()}, {"videos", videos}};
    }

    void serveMedia(int fd, const HttpRequest& request, const std::string& encodedName) const {
        const std::string name = percentDecode(encodedName);
        if (name.empty() || name == "." || name == ".." ||
            name.find('/') != std::string::npos || name.find('\\') != std::string::npos) {
            respond(fd, 404, "text/plain", "Not found");
            return;
        }
        const fs::path file = config_.videosRoot / name;
        std::error_code error;
        if (fs::is_symlink(file, error) || !fs::is_regular_file(file, error) ||
            !isSupportedMedia(file)) {
            respond(fd, 404, "text/plain", "Not found");
            return;
        }
        const auto range = request.headers.find("range");
        respondFile(fd, file, range == request.headers.end() ? std::string() : range->second);
    }

    void serveStatic(int fd, std::string path) {
        if (path.find("..") != std::string::npos) {
            respond(fd, 404, "text/plain", "Not found");
            return;
        }
        if (path.empty() || path == "/") path = "/index.html";
        fs::path file = config_.webRoot / path.substr(1);
        if (!fs::is_regular_file(file)) file = config_.webRoot / "index.html";
        if (!fs::is_regular_file(file)) {
            respond(fd, 404, "text/plain",
                    "Svelte build not found. Run: cd src/webui && npm install && npm run build\n");
            return;
        }
        respond(fd, 200, mimeType(file), readTextFile(file));
    }

    ServerConfig config_;
    EventLog events_;
    WorkerProcess worker_;
    std::mutex clientMutex_;
    std::condition_variable clientCondition_;
    std::size_t activeClients_ = 0;
    std::unordered_set<int> clientFds_;
    std::atomic<bool> idleStopping_{false};
    std::atomic<std::int64_t> lastWorkerUseMs_{0};
    std::thread idleThread_;
};

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable << " --worker PATH --engine PATH --mapping PATH --labels PATH\n"
              << "       [--host 0.0.0.0] [--port 8080] [--web-root src/webui/dist]\n"
              << "       [--videos-root /workspace/videos]\n";
}

ServerConfig parseArguments(int argc, char** argv) {
    ServerConfig config;
    config.worker.executable = "atst_sed_worker";
    config.worker.engine = "/workspace/resources/ATST-F_strong_1.trt";
    config.worker.mapping = "/workspace/class_mapping.csv";
    config.worker.labels = "/workspace/resources/ATST-F_strong_1.labels.txt";
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (index + 1 >= argc) throw std::runtime_error("Missing value after " + option);
        const std::string value = argv[++index];
        if (option == "--worker") config.worker.executable = value;
        else if (option == "--engine") config.worker.engine = value;
        else if (option == "--mapping") config.worker.mapping = value;
        else if (option == "--labels") config.worker.labels = value;
        else if (option == "--host") config.host = value;
        else if (option == "--port") config.port = std::stoi(value);
        else if (option == "--web-root") config.webRoot = value;
        else if (option == "--videos-root") config.videosRoot = value;
        else throw std::runtime_error("Unknown option: " + option);
    }
    if (config.port < 1 || config.port > 65535) throw std::runtime_error("Invalid port");
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    ::signal(SIGINT, handleSignal);
    ::signal(SIGTERM, handleSignal);
    ::signal(SIGPIPE, SIG_IGN);
    try {
        WebTestbedServer server(parseArguments(argc, argv));
        server.run();
    } catch (const std::exception& error) {
        std::cerr << "atst_sed_web_testbed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
