// config.h — experiment configuration files and the randomized trial schedule.
//
// File format (plain text, no external parser):
//
//   # comment
//   key = value                 global settings (see Config below)
//   [experiment_name]           starts an experiment block
//   kind    = chase | bandwidth
//   pattern = random | sequential          (chase only)
//   ws      = 4K, 1M, 64M                  explicit list, or
//   ws      = logspace 1K 128M 4           log2-spaced: 4 points per octave
//   stride  = 64, 16384                    (chase only; bytes, multiple of 4)
//
// Sizes accept B, K/KiB, M/MiB, G/GiB suffixes (binary multiples).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct ExperimentSpec {
    std::string name;
    std::string kind;     // "chase" or "bandwidth"
    std::string pattern;  // "random" or "sequential" (chase)
    std::vector<uint64_t> ws;
    std::vector<uint64_t> strides;
};

struct Config {
    std::string text;               // verbatim file contents (archived per session)
    std::string name = "unnamed";
    std::string phase = "smoke";    // smoke | discovery | confirmation | ...
    int reps = 2;                   // repetitions of every cell per session

    // Pointer-chase chain lengths (dependent loads per launch). For a cell
    // with S slots:
    //   n_short = max(n_min, ceil(min_traversals*S), min(ceil(target_traversals*S), n_budget))
    //   n_long  = long_ratio * n_short
    //   n_warm  = max(n_min, S)                            (one full traversal)
    // Cells whose n_long would exceed n_max are SKIPPED (reported), never
    // measured with a shorter chain, because a chain shorter than one
    // traversal measures a different (cold-cache) quantity.
    uint64_t n_min = 1ull << 16;
    uint64_t n_max = 1ull << 23;
    double min_traversals = 1.0;
    double target_traversals = 8.0;
    uint64_t n_budget = 1ull << 21;
    uint64_t long_ratio = 2;

    // Clock control. Every trial's command buffer begins with a spin-up ALU
    // chain of spin_n steps (raises the GPU clock after idle or low-utilization
    // periods), and contains ALU reference chains of alu_ref_n steps before,
    // between and after the two timed encoders (clock covariates).
    uint64_t spin_n = 1ull << 21;
    uint64_t alu_ref_n = 1ull << 16;

    // Drift sentinel: a fixed chase cell inserted every `sentinel_every` trials.
    int sentinel_every = 10;
    uint64_t sentinel_ws = 4096;
    uint64_t sentinel_stride = 64;

    // Bandwidth kernel: passes_short = max(1, ceil(bw_min_bytes / ws)),
    // passes_long = long_ratio * passes_short, one warm-up pass.
    uint32_t bw_threads = 65536;
    uint32_t bw_tg = 256;
    uint64_t bw_min_bytes = 64ull << 20;

    double session_warmup_s = 2.0;  // GPU busy time before the first trial

    std::vector<ExperimentSpec> experiments;
};

bool parse_size(const std::string& s, uint64_t& out);
bool parse_size_list(const std::string& s, std::vector<uint64_t>& out, std::string& err);
bool parse_config(const std::string& text, Config& cfg, std::string& err);

// One experimental cell = one fully specified configuration.
struct Cell {
    std::string experiment;
    std::string kind;     // chase | bandwidth | alu
    std::string pattern;  // random | sequential | linear (bandwidth) | none (alu)
    uint64_t ws_bytes = 0;      // bytes spanned by the slots actually used
    uint64_t stride_bytes = 0;
    uint64_t slots = 0;         // chase: chain elements; bandwidth: 16-byte vectors
    bool sentinel = false;
    // Iteration counts: chase = dependent loads; bandwidth = passes over the buffer.
    uint64_t n_warm = 0, n_short = 0, n_long = 0;
};

struct Trial {
    Cell cell;
    int rep = 0;
};

// Expand the configuration into its unique cells. Requested sizes are rounded
// DOWN to a whole number of slots (chase) or 16-byte vectors (bandwidth);
// combinations with fewer than 2 slots are skipped and reported in `notes`.
std::vector<Cell> expand_cells(const Config& cfg, std::vector<std::string>& notes);

// Randomized schedule: every cell × reps, shuffled (Fisher–Yates with the
// given seed), with the drift-sentinel cell inserted at the start, after every
// `sentinel_every` trials, and at the end.
std::vector<Trial> make_schedule(const Config& cfg, const std::vector<Cell>& cells,
                                 uint64_t seed);

// Fill in n_warm / n_short / n_long; false if the cell violates n_max.
bool set_iterations(const Config& cfg, Cell& c);
Cell sentinel_cell(const Config& cfg);
