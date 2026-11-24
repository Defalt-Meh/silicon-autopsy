// record.h — raw-data output (CSV), session metadata (JSON) and system information.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// Append-only CSV writer. Every row is flushed immediately so that a crash
// or interruption never loses completed measurements.
class CsvWriter {
public:
    bool open(const std::string& path, const std::vector<std::string>& header);
    void row(const std::vector<std::string>& cells);
    ~CsvWriter();
private:
    FILE* f_ = nullptr;
    size_t ncol_ = 0;
};

// Number formatting for CSV/JSON. Empty string = missing value (NA).
std::string fmt_u(uint64_t v);
std::string fmt_f(double v, int sig = 10);

std::string utc_now_iso();      // 2026-11-24T09:30:12.345Z
std::string utc_now_compact();  // 20261124T093012Z

// Minimal JSON object builder (flat values plus nested raw JSON).
class Json {
public:
    Json& str(const std::string& k, const std::string& v);
    Json& num(const std::string& k, double v);
    Json& integer(const std::string& k, long long v);
    Json& boolean(const std::string& k, bool v);
    Json& raw(const std::string& k, const std::string& json);
    std::string dump(int indent = 2) const;
private:
    std::vector<std::pair<std::string, std::string>> kv_;
};
std::string json_escape(const std::string& s);

// Host system and provenance information, recorded with every session.
struct SystemInfo {
    std::string os_version, os_build, hw_model, cpu_brand, power_source;
    uint64_t memsize = 0, page_size = 0;
    bool low_power_mode = false;
    std::string git_commit, git_dirty, host_compiler, metal_compiler;
};
SystemInfo collect_system_info();
std::string system_info_json(const SystemInfo& s);

bool write_text_file(const std::string& path, const std::string& text);
bool make_dirs(const std::string& path);
