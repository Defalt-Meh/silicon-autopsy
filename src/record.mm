// record.mm — see record.h.
#import <Foundation/Foundation.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <sys/sysctl.h>
#include <sys/time.h>

#include <cmath>
#include <ctime>
#include <sstream>

#include "record.h"

#ifndef SA_SOURCE_DIR
#define SA_SOURCE_DIR "."
#endif
#ifndef SA_METAL_VERSION
#define SA_METAL_VERSION "unknown"
#endif

// ----------------------------------------------------------------------- CSV
bool CsvWriter::open(const std::string& path, const std::vector<std::string>& header) {
    f_ = std::fopen(path.c_str(), "wx");  // "x": never overwrite an existing file
    if (!f_) return false;
    ncol_ = header.size();
    row(header);
    return true;
}

void CsvWriter::row(const std::vector<std::string>& cells) {
    if (!f_) return;
    for (size_t i = 0; i < cells.size(); ++i) {
        // Values never contain commas/quotes by construction, but quote defensively.
        const std::string& c = cells[i];
        if (c.find_first_of(",\"\n") != std::string::npos) {
            std::string q = "\"";
            for (char ch : c) { if (ch == '"') q += '"'; q += ch; }
            std::fputs((q + "\"").c_str(), f_);
        } else {
            std::fputs(c.c_str(), f_);
        }
        std::fputc(i + 1 < cells.size() ? ',' : '\n', f_);
    }
    if (cells.size() != ncol_) std::fprintf(stderr, "warning: CSV row has %zu columns, header has %zu\n", cells.size(), ncol_);
    std::fflush(f_);
}

CsvWriter::~CsvWriter() { if (f_) std::fclose(f_); }

std::string fmt_u(uint64_t v) { return std::to_string(v); }

std::string fmt_f(double v, int sig) {
    if (!std::isfinite(v)) return "";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", sig, v);
    return buf;
}

// ---------------------------------------------------------------------- time
std::string utc_now_iso() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    struct tm tm;
    gmtime_r(&tv.tv_sec, &tm);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(tv.tv_usec / 1000));
    return buf;
}

std::string utc_now_compact() {
    std::time_t t = std::time(nullptr);
    struct tm tm;
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
    return buf;
}

// ---------------------------------------------------------------------- JSON
std::string json_escape(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += static_cast<char>(c);
        }
    }
    return o + "\"";
}

Json& Json::str(const std::string& k, const std::string& v) { kv_.push_back({k, json_escape(v)}); return *this; }
Json& Json::num(const std::string& k, double v) { kv_.push_back({k, std::isfinite(v) ? fmt_f(v, 15) : "null"}); return *this; }
Json& Json::integer(const std::string& k, long long v) { kv_.push_back({k, std::to_string(v)}); return *this; }
Json& Json::boolean(const std::string& k, bool v) { kv_.push_back({k, v ? "true" : "false"}); return *this; }
Json& Json::raw(const std::string& k, const std::string& json) { kv_.push_back({k, json}); return *this; }

std::string Json::dump(int indent) const {
    const std::string pad(indent, ' ');
    std::string o = "{\n";
    for (size_t i = 0; i < kv_.size(); ++i) {
        // Indent nested objects by re-indenting their lines.
        std::string v = kv_[i].second, vi;
        for (char c : v) { vi += c; if (c == '\n') vi += pad; }
        o += pad + json_escape(kv_[i].first) + ": " + vi + (i + 1 < kv_.size() ? ",\n" : "\n");
    }
    return o + "}";
}

// ------------------------------------------------------------- system info
static std::string sysctl_str(const char* name) {
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return "";
    std::string s(len, '\0');
    if (sysctlbyname(name, s.data(), &len, nullptr, 0) != 0) return "";
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

static uint64_t sysctl_u64(const char* name) {
    uint64_t v = 0;
    size_t len = sizeof v;
    if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return 0;
    return v;
}

static std::string run_cmd(const std::string& cmd) {
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return "";
    char buf[512];
    while (std::fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

static std::string power_source() {
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (!info) return "unknown";
    CFStringRef type = IOPSGetProvidingPowerSourceType(info);
    std::string s = "unknown";
    if (type) s = [(__bridge NSString*)type UTF8String];
    CFRelease(info);
    return s;
}

SystemInfo collect_system_info() {
    SystemInfo s;
    s.os_version = [[NSProcessInfo processInfo].operatingSystemVersionString UTF8String];
    s.os_build = sysctl_str("kern.osversion");
    s.hw_model = sysctl_str("hw.model");
    s.cpu_brand = sysctl_str("machdep.cpu.brand_string");
    s.memsize = sysctl_u64("hw.memsize");
    s.page_size = sysctl_u64("hw.pagesize");
    s.power_source = power_source();
    s.low_power_mode = [NSProcessInfo processInfo].lowPowerModeEnabled;
    const std::string git = std::string("git -C '") + SA_SOURCE_DIR + "' ";
    s.git_commit = run_cmd(git + "rev-parse HEAD 2>/dev/null");
    s.git_dirty = run_cmd(git + "status --porcelain 2>/dev/null").empty() ? "clean" : "dirty";
    s.host_compiler = __VERSION__;
    s.metal_compiler = SA_METAL_VERSION;
    return s;
}

std::string system_info_json(const SystemInfo& s) {
    Json j;
    j.str("os_version", s.os_version).str("os_build", s.os_build).str("hw_model", s.hw_model)
     .str("cpu_brand", s.cpu_brand).integer("memsize_bytes", static_cast<long long>(s.memsize))
     .integer("page_size_bytes", static_cast<long long>(s.page_size)).str("power_source", s.power_source)
     .boolean("low_power_mode", s.low_power_mode).str("git_commit", s.git_commit).str("git_tree", s.git_dirty)
     .str("host_compiler", s.host_compiler).str("metal_compiler", s.metal_compiler);
    return j.dump();
}

bool write_text_file(const std::string& path, const std::string& text) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::fputs(text.c_str(), f);
    std::fputc('\n', f);
    std::fclose(f);
    return true;
}

bool make_dirs(const std::string& path) {
    NSError* e = nil;
    return [[NSFileManager defaultManager] createDirectoryAtPath:[NSString stringWithUTF8String:path.c_str()]
                                     withIntermediateDirectories:YES attributes:nil error:&e];
}
