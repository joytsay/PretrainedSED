#pragma once

#include "../common/base64.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>

namespace minitts::server {

// Each recording and its metadata commit together in one durable JSON file.
class VoiceLibrary {
public:
    explicit VoiceLibrary(std::filesystem::path root) : root_(std::move(root)) {}

    void import_reference(const std::string & name, const std::filesystem::path & wav,
                          const std::string & transcript) {
        namespace json = engine::io::json;
        const auto id = "preset-" + name;
        validate_id(id);
        const auto marker = root_ / (id + ".imported");
        if (std::filesystem::exists(marker) || !std::filesystem::is_regular_file(wav)) return;
        if (std::filesystem::file_size(wav) > 16 * 1024 * 1024) {
            throw std::invalid_argument("Reference voice exceeds the 16 MiB limit");
        }
        std::ifstream input(wav, std::ios::binary);
        if (!input) throw std::ios_base::failure("Could not read reference voice: " + wav.string());
        const std::vector<uint8_t> audio((std::istreambuf_iterator<char>(input)), {});
        if (input.bad()) throw std::ios_base::failure("Could not read reference voice: " + wav.string());
        save(json::stringify(json::Value::make_object({
            {"id", json::Value::make_string(id)}, {"name", json::Value::make_string(name)},
            {"transcript", json::Value::make_string(transcript)},
            {"createdAt", json::Value::make_number(0)},
            {"audio", json::Value::make_string(minitts::app::base64_encode(audio))}
        })));
        // Import once, so deleting a preset also survives server restarts.
        std::ofstream imported(marker);
        imported.close();
        if (!imported) throw std::ios_base::failure("Could not mark reference voice as imported");
    }

    std::string list() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string result = "{\"voices\":[";
        bool first = true;
        if (std::filesystem::exists(root_)) {
            for (const auto & entry : std::filesystem::directory_iterator(root_)) {
                if (entry.is_symlink() || !entry.is_regular_file() || entry.path().extension() != ".json") continue;
                const auto voice = engine::io::json::parse_file(entry.path());
                if (!first) result += ',';
                first = false;
                result += engine::io::json::stringify(voice);
            }
        }
        return result + "]}";
    }

    void save(const std::string & body) {
        namespace json = engine::io::json;
        // A 16 MiB WAV, including base64 and metadata overhead.
        if (body.size() > 23 * 1024 * 1024) throw std::invalid_argument("Reference voice exceeds the 16 MiB limit");
        const auto value = json::parse(body);
        const auto id = json::require_string(value, "id");
        validate_id(id);
        const auto name = json::require_string(value, "name");
        const auto transcript = json::require_string(value, "transcript");
        const auto created = json::require_i64(value, "createdAt");
        const auto audio = minitts::app::base64_decode(json::require_string(value, "audio"));
        if (name.empty() || name.size() > 256 || transcript.size() > 65536 || created < 0) {
            throw std::invalid_argument("Invalid reference voice metadata");
        }
        if (audio.size() < 44 || audio.size() > 16 * 1024 * 1024 ||
            std::string(audio.begin(), audio.begin() + 4) != "RIFF" ||
            std::string(audio.begin() + 8, audio.begin() + 12) != "WAVE") {
            throw std::invalid_argument("Reference voice must be a WAV recording of at most 16 MiB");
        }
        const auto record = json::stringify(json::Value::make_object({
            {"id", json::Value::make_string(id)}, {"name", json::Value::make_string(name)},
            {"transcript", json::Value::make_string(transcript)},
            {"createdAt", json::Value::make_number(static_cast<double>(created))},
            {"audio", json::Value::make_string(minitts::app::base64_encode(audio))}
        }));
        std::lock_guard<std::mutex> lock(mutex_);
        std::filesystem::create_directories(root_);
        const auto path = root_ / (id + ".json");
        // Idempotent imports never overwrite another client's saved recording.
        if (std::filesystem::exists(path)) return;
        const auto temporary = root_ / (id + ".tmp");
        try {
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            stream.write(record.data(), static_cast<std::streamsize>(record.size()));
            stream.close();
            if (!stream) throw std::ios_base::failure("Could not save reference voice");
            std::filesystem::rename(temporary, path);
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove(temporary, ec);
            throw;
        }
    }

    void erase(const std::string & id) {
        validate_id(id);
        std::lock_guard<std::mutex> lock(mutex_);
        std::filesystem::remove(root_ / (id + ".json"));
    }

private:
    static void validate_id(const std::string & id) {
        if (id.empty() || id.size() > 128 || !std::all_of(id.begin(), id.end(), [](unsigned char ch) {
            return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
        })) throw std::invalid_argument("Invalid reference voice ID");
    }
    std::filesystem::path root_;
    std::mutex mutex_;
};

} // namespace minitts::server
