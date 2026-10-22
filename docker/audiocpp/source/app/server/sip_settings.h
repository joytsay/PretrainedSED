#pragma once

#include "server_protocol.h"
#include "engine/framework/io/json.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

namespace minitts::server {

// Shared with the SIP container. Values are data parsed by sipEnv.h, never shell code.
inline HttpResponse sip_settings_response(const HttpRequest & request, bool management,
                                         const std::filesystem::path & base) {
    if (!management) return error_response(403, "SIP configuration is disabled", "forbidden");
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const char * configured = std::getenv("SIP_SETTINGS_DIR");
    const auto root = configured && *configured ? std::filesystem::path(configured) : base / "sip-data";
    const auto path = root / "gateway.env";
    namespace json = engine::io::json;
    if (request.method == "GET") {
        std::string proxy = "sip:175.180.90.33:9900", account = "sip:521@175.180.90.33";
        std::string auth = "521", display = "521";
        bool has_secret = false;
        std::ifstream input(path);
        std::string line;
        while (std::getline(input, line)) {
            const auto equals = line.find('=');
            if (equals == std::string::npos) continue;
            const auto key = line.substr(0, equals);
            const auto value = json::parse(line.substr(equals + 1)).as_string();
            if (key == "SIP_PROXY") proxy = value;
            else if (key == "SIP_ACCOUNT") account = value;
            else if (key == "SIP_AUTHUSER") auth = value;
            else if (key == "SIP_DISPLAY_NAME") display = value;
            else if (key == "SIP_SECRET") has_secret = !value.empty();
        }
        // A heartbeat describes the supervisor, not successful PBX registration.
        std::error_code ec;
        const auto heartbeat = std::filesystem::last_write_time(root / "worker.heartbeat", ec);
        const bool worker = !ec && std::filesystem::file_time_type::clock::now() - heartbeat < std::chrono::seconds(10);
        auto response = json_response("{\"proxy\":" + json_quote(proxy) + ",\"account\":" + json_quote(account) +
            ",\"auth_user\":" + json_quote(auth) + ",\"display_name\":" + json_quote(display) +
            ",\"has_secret\":" + (has_secret ? "true" : "false") + ",\"worker_running\":" + (worker ? "true" : "false") + "}");
        response.headers["Cache-Control"] = "no-store";
        return response;
    }
    if (request.body.size() > 8192) return error_response(413, "SIP settings are too large", "invalid_request_error");
    const auto body = json::parse(request.body);
    auto proxy = json::require_string(body, "proxy"), account = json::require_string(body, "account");
    auto auth = json::require_string(body, "auth_user"), display = json::require_string(body, "display_name");
    auto secret = json::require_string(body, "secret");
    auto safe = [](const std::string & value) {
        return value.size() <= 1024 && value.find_first_of("\r\n") == std::string::npos && value.find('\0') == std::string::npos;
    };
    if (!safe(proxy) || !safe(account) || !safe(auth) || !safe(display) || !safe(secret) ||
        proxy.rfind("sip:", 0) != 0 || account.rfind("sip:", 0) != 0 || account.find('@') == std::string::npos ||
        proxy.find_first_of(" \t\"<>") != std::string::npos || account.find_first_of(" \t\"<>") != std::string::npos ||
        auth.empty() || display.find_first_of("\"\\<>") != std::string::npos) {
        return error_response(400, "Enter valid SIP URIs, authentication username, and display name", "invalid_request_error");
    }
    // An empty password keeps the existing secret, so GET never needs to expose it.
    if (secret.empty()) {
        std::ifstream input(path);
        std::string line;
        while (std::getline(input, line)) {
            if (line.rfind("SIP_SECRET=", 0) == 0) secret = json::parse(line.substr(11)).as_string();
        }
    }
    if (secret.empty()) return error_response(400, "Enter the SIP secret", "invalid_request_error");
    std::filesystem::create_directories(root);
    const auto temporary = root / "gateway.env.tmp";
    {
        // Create with private permissions before writing credentials.
        std::ofstream create(temporary, std::ios::trunc);
        if (!create) throw std::runtime_error("Cannot create SIP settings");
        std::filesystem::permissions(temporary, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
        create << "SIP_PROXY=" << json_quote(proxy) << '\n' << "SIP_ACCOUNT=" << json_quote(account) << '\n'
            << "SIP_AUTHUSER=" << json_quote(auth) << '\n' << "SIP_DISPLAY_NAME=" << json_quote(display) << '\n'
            << "SIP_SECRET=" << json_quote(secret) << '\n' << "SIP_TARGET=\"listen\"\n";
        create.close();
        if (!create) throw std::runtime_error("Cannot write SIP settings");
    }
    std::filesystem::rename(temporary, path);
    return json_response("{\"ok\":true,\"message\":\"Settings saved. The SIP worker will reload within two seconds; an active call will end.\"}");
}

} // namespace minitts::server
