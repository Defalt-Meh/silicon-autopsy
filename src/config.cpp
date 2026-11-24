// config.cpp — see config.h.
#include "config.h"

#include <cctype>
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "rng.h"

static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool parse_size(const std::string& raw, uint64_t& out) {
    const std::string s = lower(trim(raw));
    size_t pos = 0;
    while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.')) ++pos;
    if (pos == 0) return false;
    const double num = std::stod(s.substr(0, pos));
    const std::string suf = trim(s.substr(pos));
    double mul = 1;
    if (suf.empty() || suf == "b") mul = 1;
    else if (suf == "k" || suf == "kib") mul = 1024.0;
    else if (suf == "m" || suf == "mib") mul = 1024.0 * 1024.0;
    else if (suf == "g" || suf == "gib") mul = 1024.0 * 1024.0 * 1024.0;
    else return false;
    out = static_cast<uint64_t>(std::llround(num * mul));
    return true;
}

bool parse_size_list(const std::string& raw, std::vector<uint64_t>& out, std::string& err) {
    out.clear();
    const std::string s = trim(raw);
    if (lower(s).rfind("logspace", 0) == 0) {
        // logspace LO HI PPO: sizes 2^(log2(LO) + j/PPO) for j = 0,1,... up to HI.
        std::istringstream is(s.substr(8));
        std::string lo_s, hi_s;
        double ppo = 0;
        if (!(is >> lo_s >> hi_s >> ppo) || ppo <= 0) {
            err = "logspace expects: logspace LO HI POINTS_PER_OCTAVE";
            return false;
        }
        uint64_t lo = 0, hi = 0;
        if (!parse_size(lo_s, lo) || !parse_size(hi_s, hi) || lo == 0 || hi < lo) {
            err = "bad logspace bounds";
            return false;
        }
        const double e0 = std::log2(static_cast<double>(lo));
        const double e1 = std::log2(static_cast<double>(hi));
        for (int j = 0;; ++j) {
            const double e = e0 + j / ppo;
            if (e > e1 + 1e-9) break;
            out.push_back(static_cast<uint64_t>(std::llround(std::exp2(e))));
        }
        return true;
    }
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        uint64_t v = 0;
        if (!parse_size(item, v)) {
            err = "bad size: '" + trim(item) + "'";
            return false;
        }
        out.push_back(v);
    }
    if (out.empty()) {
        err = "empty size list";
        return false;
    }
    return true;
}

bool parse_config(const std::string& text, Config& cfg, std::string& err) {
    cfg.text = text;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    ExperimentSpec* cur = nullptr;
    while (std::getline(in, line)) {
        ++lineno;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        const std::string where = "line " + std::to_string(lineno) + ": ";

        if (line.front() == '[') {
            if (line.back() != ']') { err = where + "unterminated section"; return false; }
            cfg.experiments.push_back(ExperimentSpec{});
            cur = &cfg.experiments.back();
            cur->name = trim(line.substr(1, line.size() - 2));
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) { err = where + "expected key = value"; return false; }
        const std::string key = lower(trim(line.substr(0, eq)));
        const std::string val = trim(line.substr(eq + 1));

        try {
            if (cur) {  // experiment-level keys
                if (key == "kind") cur->kind = lower(val);
                else if (key == "pattern") cur->pattern = lower(val);
                else if (key == "ws") { if (!parse_size_list(val, cur->ws, err)) { err = where + err; return false; } }
                else if (key == "stride") { if (!parse_size_list(val, cur->strides, err)) { err = where + err; return false; } }
                else { err = where + "unknown experiment key '" + key + "'"; return false; }
                continue;
            }
            uint64_t sz = 0;
            auto size_val = [&](uint64_t& dst) {
                if (!parse_size(val, sz)) throw std::runtime_error("bad size");
                dst = sz;
            };
            if (key == "name") cfg.name = val;
            else if (key == "phase") cfg.phase = lower(val);
            else if (key == "reps") cfg.reps = std::stoi(val);
            else if (key == "n_min") size_val(cfg.n_min);
            else if (key == "n_max") size_val(cfg.n_max);
            else if (key == "min_traversals") cfg.min_traversals = std::stod(val);
            else if (key == "target_traversals") cfg.target_traversals = std::stod(val);
            else if (key == "n_budget") size_val(cfg.n_budget);
            else if (key == "long_ratio") size_val(cfg.long_ratio);
            else if (key == "alu_ref_n") size_val(cfg.alu_ref_n);
            else if (key == "spin_n") size_val(cfg.spin_n);
            else if (key == "sentinel_every") cfg.sentinel_every = std::stoi(val);
            else if (key == "sentinel_ws") size_val(cfg.sentinel_ws);
            else if (key == "sentinel_stride") size_val(cfg.sentinel_stride);
            else if (key == "bw_threads") { uint64_t v; size_val(v); cfg.bw_threads = static_cast<uint32_t>(v); }
            else if (key == "bw_tg") { uint64_t v; size_val(v); cfg.bw_tg = static_cast<uint32_t>(v); }
            else if (key == "bw_min_bytes") size_val(cfg.bw_min_bytes);
            else if (key == "session_warmup_s") cfg.session_warmup_s = std::stod(val);
            else { err = where + "unknown key '" + key + "'"; return false; }
        } catch (const std::exception&) {
            err = where + "bad value for '" + key + "'";
            return false;
        }
    }

    // Semantic checks.
    if (cfg.reps < 1) { err = "reps must be >= 1"; return false; }
    if (cfg.n_min < 1 || cfg.n_max < cfg.n_min || cfg.n_max > 0xFFFFFFFFull) { err = "need 1 <= n_min <= n_max < 2^32"; return false; }
    if (cfg.min_traversals < 1.0) { err = "min_traversals must be >= 1"; return false; }
    if (cfg.target_traversals < cfg.min_traversals) { err = "target_traversals must be >= min_traversals"; return false; }
    if (cfg.long_ratio < 2) { err = "long_ratio must be >= 2"; return false; }
    if (cfg.alu_ref_n < 1 || cfg.alu_ref_n > 0xFFFFFFFFull) { err = "bad alu_ref_n"; return false; }
    if (cfg.spin_n > 0xFFFFFFFFull) { err = "bad spin_n"; return false; }
    if (cfg.sentinel_stride % 4 != 0 || cfg.sentinel_ws / cfg.sentinel_stride < 2) { err = "bad sentinel ws/stride"; return false; }
    if (cfg.bw_tg == 0 || cfg.bw_threads % cfg.bw_tg != 0) { err = "bw_threads must be a multiple of bw_tg"; return false; }
    for (const auto& e : cfg.experiments) {
        const std::string w = "experiment [" + e.name + "]: ";
        if (e.kind != "chase" && e.kind != "bandwidth") { err = w + "kind must be chase or bandwidth"; return false; }
        if (e.ws.empty()) { err = w + "missing ws"; return false; }
        if (e.kind == "chase") {
            if (e.pattern != "random" && e.pattern != "sequential") { err = w + "pattern must be random or sequential"; return false; }
            if (e.strides.empty()) { err = w + "missing stride"; return false; }
            for (uint64_t s : e.strides)
                if (s < 4 || s % 4 != 0) { err = w + "stride must be a positive multiple of 4 bytes"; return false; }
        }
    }
    return true;
}

std::vector<Cell> expand_cells(const Config& cfg, std::vector<std::string>& notes) {
    std::vector<Cell> cells;
    std::set<std::tuple<std::string, uint64_t, uint64_t>> seen;
    for (const auto& e : cfg.experiments) {
        if (e.kind == "chase") {
            for (uint64_t stride : e.strides) {
                for (uint64_t ws_req : e.ws) {
                    const uint64_t slots = ws_req / stride;
                    if (slots < 2) {
                        notes.push_back("[" + e.name + "] skipped ws=" + std::to_string(ws_req) +
                                        " stride=" + std::to_string(stride) + " (fewer than 2 slots)");
                        continue;
                    }
                    const uint64_t ws = slots * stride;
                    if (ws / 4 > 0xFFFFFFFFull) {
                        notes.push_back("[" + e.name + "] skipped ws=" + std::to_string(ws) + " (exceeds 32-bit word index)");
                        continue;
                    }
                    if (!seen.insert({e.name, ws, stride}).second) continue;  // rounding duplicate
                    Cell c{e.name, "chase", e.pattern, ws, stride, slots, false};
                    if (!set_iterations(cfg, c)) {
                        notes.push_back("[" + e.name + "] skipped ws=" + std::to_string(ws) + " stride=" +
                                        std::to_string(stride) + " (full traversal needs n_long=" +
                                        std::to_string(c.n_long) + " > n_max)");
                        continue;
                    }
                    cells.push_back(c);
                }
            }
        } else {  // bandwidth
            for (uint64_t ws_req : e.ws) {
                const uint64_t ws = ws_req / 16 * 16;
                if (ws < 16) continue;
                if (!seen.insert({e.name, ws, 16}).second) continue;
                Cell c{e.name, "bandwidth", "linear", ws, 16, ws / 16, false};
                set_iterations(cfg, c);
                cells.push_back(c);
            }
        }
    }
    return cells;
}

bool set_iterations(const Config& cfg, Cell& c) {
    if (c.kind == "chase") {
        // Short launch: at least n_min loads and min_traversals full traversals
        // (hard requirement), and up to target_traversals traversals while
        // that stays within n_budget loads. More traversals in the timed
        // difference shrink the error from an unmatched start-up transient
        // (docs/measurement.md §3.5) in proportion.
        const double s = static_cast<double>(c.slots);
        const uint64_t hard = static_cast<uint64_t>(std::ceil(cfg.min_traversals * s));
        const uint64_t target = std::min(static_cast<uint64_t>(std::ceil(cfg.target_traversals * s)), cfg.n_budget);
        c.n_short = std::max({cfg.n_min, hard, target});
        c.n_long = cfg.long_ratio * c.n_short;
        c.n_warm = std::max(cfg.n_min, c.slots);  // one full traversal
        return c.n_long <= cfg.n_max;
    }
    // bandwidth: iteration counts are passes over the whole buffer
    c.n_short = std::max<uint64_t>(1, (cfg.bw_min_bytes + c.ws_bytes - 1) / c.ws_bytes);
    c.n_long = cfg.long_ratio * c.n_short;
    c.n_warm = 1;
    return c.n_long < (1ull << 31);
}

Cell sentinel_cell(const Config& cfg) {
    const uint64_t slots = cfg.sentinel_ws / cfg.sentinel_stride;
    Cell c{"sentinel_chase", "chase", "random", slots * cfg.sentinel_stride, cfg.sentinel_stride, slots, true};
    set_iterations(cfg, c);
    return c;
}

std::vector<Trial> make_schedule(const Config& cfg, const std::vector<Cell>& cells,
                                 uint64_t seed) {
    std::vector<Trial> pool;
    for (int r = 0; r < cfg.reps; ++r)
        for (const auto& c : cells) pool.push_back(Trial{c, r});

    // Fisher–Yates shuffle: a uniformly random permutation of the trials, so
    // slow drifts (temperature, background load) are not confounded with any
    // particular cell.
    Rng rng(seed);
    for (size_t i = pool.size(); i > 1; --i) {
        const size_t j = static_cast<size_t>(rng.below(i));
        std::swap(pool[i - 1], pool[j]);
    }

    std::vector<Trial> out;
    const Cell sc = sentinel_cell(cfg);
    int sent_rep = 0;
    auto add_sentinel = [&] { out.push_back(Trial{sc, sent_rep++}); };
    add_sentinel();
    for (size_t i = 0; i < pool.size(); ++i) {
        out.push_back(pool[i]);
        if (cfg.sentinel_every > 0 && (i + 1) % cfg.sentinel_every == 0) add_sentinel();
    }
    if (cfg.sentinel_every > 0 && pool.size() % cfg.sentinel_every != 0) add_sentinel();
    return out;
}
