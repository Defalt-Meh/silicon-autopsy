// main.mm — SILICON AUTOPSY command-line interface and measurement procedures.
//
//   silicon-autopsy info                          device capabilities + clock calibration
//   silicon-autopsy validate [options]            measurement-validation protocol
//   silicon-autopsy benchmark [options]           run one experimental session
//
// Options:
//   --config PATH      experiment configuration (benchmark; default configs/smoke.cfg)
//   --seed N           session seed (default: drawn from the OS entropy source, recorded)
//   --session-id ID    session identifier (default: UTC timestamp)
//   --out DIR          output root (default: data/raw for benchmark, data/validation for validate)
//   --dry-run          print the schedule summary and exit (benchmark)
//
// See docs/measurement.md for the precise definition of every recorded quantity.

#import "metal_ctx.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../kernels/shared_types.h"
#include "config.h"
#include "record.h"
#include "workloads.h"

#ifndef SA_METALLIB_PATH
#error "SA_METALLIB_PATH must be defined by the build"
#endif
#ifndef SA_SOURCE_DIR
#define SA_SOURCE_DIR "."
#endif

static const uint32_t kUnwritten = 0xFFFFFFFFu;  // preset result word: a kernel that never ran is detected

// =========================================================================
// Dispatch builders
// =========================================================================
static Dispatch chase_dispatch(MetalCtx& ctx, id<MTLBuffer> arena, uint32_t slot, uint32_t start, uint64_t n) {
    Dispatch d;
    d.pso = ctx.pso_chase;
    d.b0 = arena;
    d.out_slot = slot;
    d.args = as_bytes(ChaseArgs{start, static_cast<uint32_t>(n)});
    return d;
}

static Dispatch alu_dispatch(MetalCtx& ctx, uint32_t slot, uint32_t seed, uint64_t n) {
    Dispatch d;
    d.pso = ctx.pso_alu;
    d.out_slot = slot;
    d.args = as_bytes(AluArgs{seed, static_cast<uint32_t>(n)});
    return d;
}

static Dispatch bw_dispatch(MetalCtx& ctx, id<MTLBuffer> arena, uint32_t slot, uint64_t n_vec, uint64_t passes,
                            uint32_t threads, uint32_t tg) {
    Dispatch d;
    d.pso = ctx.pso_bw;
    d.b0 = arena;
    d.out_slot = slot;
    d.args = as_bytes(BwArgs{static_cast<uint32_t>(n_vec), static_cast<uint32_t>(passes)});
    d.grid = MTLSizeMake(threads, 1, 1);
    d.tg = MTLSizeMake(tg, 1, 1);
    return d;
}

// =========================================================================
// One trial = ONE command buffer with seven encoders:
//
//   [0] clock spin-up            (spin_n ALU steps; untimed purpose)
//   [1] ALU clock reference      (alu_ref_n dependent integer steps)
//   [2] warm-up                  (chase: n_warm = n_short loads, >= 1 full traversal;
//                                 bandwidth: 1 pass)
//   [3] short or long            (order randomized per trial)
//   [4] ALU clock reference
//   [5] long or short
//   [6] ALU clock reference
//
// The spin-up exists because validation showed the GPU clock is often low at
// the start of a trial that follows a long, latency-bound (low-utilization)
// trial. The three references measure the clock around each timed encoder.
//
// Encoders [2] and [3] are each preceded by at least one full traversal of
// the same buffer in the same command buffer, so they start from the same
// cache state. Each also begins with the same transient (state carried
// across encoders is not identical to state within an encoder; see
// docs/measurement.md §3). The model for an encoder of n iterations is
//     T(n) = a + b·n      for n >= one traversal,
// and the differential estimator (T_long − T_short)/(n_long − n_short)
// estimates the steady-state per-iteration cost b while cancelling a.
// =========================================================================
struct TrialResult {
    uint32_t threads = 1, tg = 1;
    bool long_first = false;
    SequenceTiming seq;
    double gpu_short = NAN, gpu_long = NAN, gpu_warm = NAN;  // ns
    double alu0 = NAN, alu1 = NAN, alu2 = NAN;               // ns per ALU step (start, mid, end)
    double bytes_short = NAN, bytes_long = NAN;
    bool valid = false;
    std::string note;
};

static void unpack(const Config& cfg, TrialResult& r) {
    if (!r.seq.ok) return;
    const auto& g = r.seq.gpu_ns;
    const double n = static_cast<double>(cfg.alu_ref_n);
    r.alu0 = g[1] / n;
    r.gpu_warm = g[2];
    r.gpu_short = r.long_first ? g[5] : g[3];
    r.gpu_long = r.long_first ? g[3] : g[5];
    r.alu1 = g[4] / n;
    r.alu2 = g[6] / n;
}

// Validate the spin-up (slot 0) and the three ALU references (slots 1, 4, 6).
static bool check_alu(MetalCtx& ctx, uint32_t seed, const Config& cfg, std::string& note) {
    const uint32_t ref = cpu_lcg(seed, cfg.alu_ref_n);
    const bool ok = ctx.out_word(0)[0] == cpu_lcg(seed, cfg.spin_n) && ctx.out_word(1)[0] == ref &&
                    ctx.out_word(4)[0] == ref && ctx.out_word(6)[0] == ref;
    if (!ok) note += "alu mismatch; ";
    return ok;
}

static TrialResult run_chase_trial(MetalCtx& ctx, id<MTLBuffer> arena, uint32_t* words, const Config& cfg,
                                   const Cell& c, uint64_t seed) {
    TrialResult r;
    Rng rng(seed);
    build_chain(words, c.slots, c.stride_bytes / 4,
                c.pattern == "sequential" ? Pattern::Sequential : Pattern::Random, rng);
    r.long_first = rng.below(2) == 1;
    const uint32_t alu_seed = static_cast<uint32_t>(rng.next());
    const uint64_t na = r.long_first ? c.n_long : c.n_short, nb = r.long_first ? c.n_short : c.n_long;

    // Every chase encoder starts at slot 0. The preceding encoder has
    // traversed the full cycle at least once, so the start position does not
    // change the cache state it sees.
    for (uint32_t s = 0; s < 7; ++s) ctx.out_word(s)[0] = kUnwritten;
    r.seq = ctx.run({alu_dispatch(ctx, 0, alu_seed, cfg.spin_n),
                     alu_dispatch(ctx, 1, alu_seed, cfg.alu_ref_n),
                     chase_dispatch(ctx, arena, 2, 0, c.n_warm),
                     chase_dispatch(ctx, arena, 3, 0, na),
                     alu_dispatch(ctx, 4, alu_seed, cfg.alu_ref_n),
                     chase_dispatch(ctx, arena, 5, 0, nb),
                     alu_dispatch(ctx, 6, alu_seed, cfg.alu_ref_n)});

    // Bit-exact validation of every encoder against the CPU reference.
    bool ok = r.seq.ok && check_alu(ctx, alu_seed, cfg, r.note);
    if (!r.seq.ok) r.note = "launch error: " + r.seq.error;
    if (ok) {
        const uint32_t end_short = cpu_chase(words, 0, c.n_short);
        const uint32_t end_long = cpu_chase(words, end_short, c.n_long - c.n_short);
        if (ctx.out_word(2)[0] != cpu_chase(words, 0, c.n_warm)) { ok = false; r.note += "warm-up chase mismatch; "; }
        if (ctx.out_word(3)[0] != (r.long_first ? end_long : end_short) ||
            ctx.out_word(5)[0] != (r.long_first ? end_short : end_long)) { ok = false; r.note += "chase mismatch; "; }
    }
    r.valid = ok;
    unpack(cfg, r);
    return r;
}

static TrialResult run_bw_trial(MetalCtx& ctx, id<MTLBuffer> arena, uint32_t* words, const Config& cfg,
                                const Cell& c, uint64_t seed) {
    TrialResult r;
    Rng rng(seed);
    fill_random(words, c.ws_bytes / 4, rng);
    r.threads = cfg.bw_threads;
    r.tg = cfg.bw_tg;
    r.long_first = rng.below(2) == 1;
    const uint32_t alu_seed = static_cast<uint32_t>(rng.next());
    const uint64_t n_vec = c.slots;
    r.bytes_short = static_cast<double>(c.n_short) * c.ws_bytes;
    r.bytes_long = static_cast<double>(c.n_long) * c.ws_bytes;
    const uint64_t pa = r.long_first ? c.n_long : c.n_short, pb = r.long_first ? c.n_short : c.n_long;

    for (uint32_t s : {0u, 1u, 4u, 6u}) ctx.out_word(s)[0] = kUnwritten;
    for (uint32_t s : {2u, 3u, 5u}) ctx.out_word(s)[0] = 0;  // atomic accumulators
    r.seq = ctx.run({alu_dispatch(ctx, 0, alu_seed, cfg.spin_n),
                     alu_dispatch(ctx, 1, alu_seed, cfg.alu_ref_n),
                     bw_dispatch(ctx, arena, 2, n_vec, c.n_warm, r.threads, r.tg),
                     bw_dispatch(ctx, arena, 3, n_vec, pa, r.threads, r.tg),
                     alu_dispatch(ctx, 4, alu_seed, cfg.alu_ref_n),
                     bw_dispatch(ctx, arena, 5, n_vec, pb, r.threads, r.tg),
                     alu_dispatch(ctx, 6, alu_seed, cfg.alu_ref_n)});

    bool ok = r.seq.ok && check_alu(ctx, alu_seed, cfg, r.note);
    if (!r.seq.ok) r.note = "launch error: " + r.seq.error;
    if (ok) {
        const uint32_t s_w = cpu_bw_reference(words, n_vec, static_cast<uint32_t>(c.n_warm), r.threads);
        const uint32_t s_a = cpu_bw_reference(words, n_vec, static_cast<uint32_t>(pa), r.threads);
        const uint32_t s_b = cpu_bw_reference(words, n_vec, static_cast<uint32_t>(pb), r.threads);
        if (ctx.out_word(2)[0] != s_w || ctx.out_word(3)[0] != s_a || ctx.out_word(5)[0] != s_b) {
            ok = false;
            r.note += "bandwidth checksum mismatch; ";
        }
    }
    r.valid = ok;
    unpack(cfg, r);
    return r;
}

// GPU idle gap (ns) between the end of encoder k-1 and the start of encoder k.
// A gap much larger than the normal inter-encoder overhead indicates that
// other GPU work (another process) was scheduled between our encoders.
static double gap(const SequenceTiming& t, size_t k) {
    if (!t.ok || k == 0 || k >= t.tick_start.size()) return NAN;
    return (static_cast<double>(t.tick_start[k]) - static_cast<double>(t.tick_end[k - 1])) * t.ns_per_tick;
}

static double max_gap(const SequenceTiming& t) {
    double m = NAN;
    for (size_t k = 1; k < t.tick_start.size(); ++k) m = std::isnan(m) ? gap(t, k) : std::max(m, gap(t, k));
    return m;
}

// =========================================================================
// CLI helpers
// =========================================================================
struct Args {
    std::string cmd, config, out, session_id;
    uint64_t seed = 0;
    bool has_seed = false, dry_run = false;
};

static void usage() {
    std::fprintf(stderr,
                 "usage: silicon-autopsy {info|validate|benchmark} [--config PATH] [--seed N]\n"
                 "                       [--session-id ID] [--out DIR] [--dry-run]\n");
}

static bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](std::string& dst) { if (i + 1 >= argc) return false; dst = argv[++i]; return true; };
        std::string v;
        if (k == "--config") { if (!need(a.config)) return false; }
        else if (k == "--out") { if (!need(a.out)) return false; }
        else if (k == "--session-id") { if (!need(a.session_id)) return false; }
        else if (k == "--seed") { if (!need(v)) return false; a.seed = std::stoull(v); a.has_seed = true; }
        else if (k == "--dry-run") a.dry_run = true;
        else return false;
    }
    return true;
}

static uint64_t entropy_seed() {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool init_ctx(MetalCtx& ctx, ClockCalibration& cal) {
    std::string err;
    if (!ctx.init(SA_METALLIB_PATH, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return false; }
    cal = ctx.calibrate(0.2);
    if (!(cal.ns_per_tick > 0)) { std::fprintf(stderr, "error: GPU clock calibration failed\n"); return false; }
    ctx.ns_per_tick = cal.ns_per_tick;
    return true;
}

static id<MTLBuffer> alloc_arena(MetalCtx& ctx, uint64_t bytes, std::string& err) {
    if (bytes > ctx.device.maxBufferLength || bytes > ctx.device.recommendedMaxWorkingSetSize / 2) {
        err = "working set too large for this device";
        return nil;
    }
    id<MTLBuffer> b = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) err = "buffer allocation failed";
    return b;
}

static std::string calib_json(const ClockCalibration& c) {
    Json j;
    j.num("ns_per_tick", c.ns_per_tick).num("interval_s", c.interval_s)
     .integer("gpu_ticks", static_cast<long long>(c.gpu_ticks)).integer("uptime_ns", static_cast<long long>(c.uptime_ns));
    return j.dump();
}

static std::string device_json(MetalCtx& ctx) {
    id<MTLDevice> d = ctx.device;
    Json j;
    j.str("name", d.name.UTF8String)
     .integer("registry_id", static_cast<long long>(d.registryID))
     .boolean("unified_memory", d.hasUnifiedMemory)
     .integer("recommended_max_working_set_bytes", static_cast<long long>(d.recommendedMaxWorkingSetSize))
     .integer("max_buffer_length", static_cast<long long>(d.maxBufferLength))
     .integer("max_threadgroup_memory", static_cast<long long>(d.maxThreadgroupMemoryLength))
     .boolean("counter_sampling_stage_boundary", [d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
     .boolean("counter_sampling_dispatch_boundary", [d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary])
     .integer("bw_thread_execution_width", static_cast<long long>(ctx.pso_bw.threadExecutionWidth));
    return j.dump();
}

static void warn_conditions(const SystemInfo& si) {
    if (si.power_source != "AC Power")
        std::fprintf(stderr, "WARNING: running on '%s'. Real data should be collected on AC power.\n", si.power_source.c_str());
    if (si.low_power_mode)
        std::fprintf(stderr, "WARNING: Low Power Mode is enabled. Disable it for real data collection.\n");
    if (si.git_dirty == "dirty")
        std::fprintf(stderr, "WARNING: git working tree is dirty; commit code before collecting real data.\n");
}

// Keep the GPU busy for `seconds` before measuring, so clocks leave idle state.
static void session_warmup(MetalCtx& ctx, double seconds) {
    const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    while ((clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) < seconds * 1e9)
        ctx.run({alu_dispatch(ctx, 0, 1, 1u << 20)});
}

// =========================================================================
// info
// =========================================================================
static int cmd_info() {
    MetalCtx ctx;
    ClockCalibration cal;
    if (!init_ctx(ctx, cal)) return 1;
    const SystemInfo si = collect_system_info();
    std::printf("device = %s\n", device_json(ctx).c_str());
    std::printf("clock_calibration = %s\n", calib_json(cal).c_str());
    std::printf("system = %s\n", system_info_json(si).c_str());
    std::printf("metallib = %s\n", SA_METALLIB_PATH);
    return 0;
}

// =========================================================================
// benchmark
// =========================================================================
static int cmd_benchmark(const Args& args) {
    const std::string cfg_path = args.config.empty() ? std::string(SA_SOURCE_DIR) + "/configs/smoke.cfg" : args.config;
    Config cfg;
    std::string err;
    const std::string text = read_file(cfg_path);
    if (text.empty()) { std::fprintf(stderr, "error: cannot read config %s\n", cfg_path.c_str()); return 1; }
    if (!parse_config(text, cfg, err)) { std::fprintf(stderr, "config error: %s\n", err.c_str()); return 1; }

    std::vector<std::string> notes;
    const std::vector<Cell> cells = expand_cells(cfg, notes);
    const uint64_t seed = args.has_seed ? args.seed : entropy_seed();
    const std::vector<Trial> schedule = make_schedule(cfg, cells, derive_seed(seed, 0));

    uint64_t max_ws = sentinel_cell(cfg).ws_bytes;
    double chase_loads = 0, bw_bytes = 0;
    for (const auto& t : schedule) {
        max_ws = std::max(max_ws, t.cell.ws_bytes);
        if (t.cell.kind == "chase") chase_loads += t.cell.n_warm + t.cell.n_short + t.cell.n_long;
        else bw_bytes += static_cast<double>(t.cell.n_warm + t.cell.n_short + t.cell.n_long) * t.cell.ws_bytes;
    }
    std::printf("config %s (%s): %zu cells x %d reps -> %zu trials incl. sentinels; arena %.1f MiB\n"
                "  GPU work: %.3g dependent loads, %.3g bytes streamed\n",
                cfg.name.c_str(), cfg.phase.c_str(), cells.size(), cfg.reps, schedule.size(), max_ws / 1048576.0,
                chase_loads, bw_bytes);
    for (const auto& n : notes) std::printf("  note: %s\n", n.c_str());
    if (args.dry_run) return 0;

    MetalCtx ctx;
    ClockCalibration cal0;
    if (!init_ctx(ctx, cal0)) return 1;
    id<MTLBuffer> arena = alloc_arena(ctx, max_ws, err);
    if (!arena) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    uint32_t* words = static_cast<uint32_t*>(arena.contents);

    const SystemInfo si = collect_system_info();
    warn_conditions(si);

    const std::string session_id = args.session_id.empty() ? utc_now_compact() : args.session_id;
    const std::string out_dir = (args.out.empty() ? std::string(SA_SOURCE_DIR) + "/data/raw" : args.out) + "/" + cfg.phase;
    make_dirs(out_dir);
    const std::string csv_path = out_dir + "/" + session_id + ".csv";
    const std::string json_path = out_dir + "/" + session_id + ".json";

    const std::vector<std::string> header = {
        "experiment_id", "session_id", "phase", "run_index", "timestamp_utc", "experiment", "kind",
        "access_pattern", "is_sentinel", "rep", "working_set_bytes", "stride_bytes", "n_slots",
        "thread_count", "threadgroup_size", "iteration_count", "iteration_count_short", "warmup_iterations",
        "long_first", "gpu_ns_short", "gpu_ns_long", "gpu_ns_warmup", "alu_ns_per_op_start", "alu_ns_per_op_mid",
        "alu_ns_per_op_end",
        "cmdbuf_ns_trial", "cpu_wall_ns_trial", "gpu_tick_trial_start", "gpu_gap_ns_before_a",
        "gpu_gap_ns_before_b", "gpu_gap_ns_max", "gpu_execution_ns",
        "effective_latency_ns", "short_latency_ns", "differential_latency_ns", "bytes_short", "bytes_long",
        "bandwidth_gbps", "short_bandwidth_gbps", "differential_bandwidth_gbps",
        "validation_ok", "validation_note", "thermal_state", "trial_seed"};
    CsvWriter csv;
    if (!csv.open(csv_path, header)) { std::fprintf(stderr, "error: cannot create %s (exists?)\n", csv_path.c_str()); return 1; }

    const std::string start_time = utc_now_iso();
    const std::string thermal_start = thermal_state_name();
    auto write_meta = [&](const std::string& status, size_t done, size_t invalid, const ClockCalibration* cal1) {
        Json j;
        j.str("session_id", session_id).str("status", status).str("phase", cfg.phase).str("config_name", cfg.name)
         .str("config_path", cfg_path).str("config_text", cfg.text)
         .str("session_seed", std::to_string(seed)).str("start_utc", start_time)
         .str("end_utc", status == "running" ? "" : utc_now_iso())
         .integer("trials_planned", static_cast<long long>(schedule.size())).integer("trials_done", static_cast<long long>(done))
         .integer("trials_invalid", static_cast<long long>(invalid)).str("thermal_state_start", thermal_start)
         .str("thermal_state_end", status == "running" ? "" : thermal_state_name())
         .num("session_warmup_s", cfg.session_warmup_s)
         .raw("clock_calibration_start", calib_json(cal0)).raw("clock_calibration_end", cal1 ? calib_json(*cal1) : "null")
         .raw("device", device_json(ctx)).raw("system", system_info_json(si))
         .str("metallib", SA_METALLIB_PATH);
        write_text_file(json_path, j.dump());
    };
    write_meta("running", 0, 0, nullptr);

    session_warmup(ctx, cfg.session_warmup_s);

    size_t invalid = 0;
    for (size_t i = 0; i < schedule.size(); ++i) {
        const Trial& tr = schedule[i];
        const Cell& c = tr.cell;
        const uint64_t tseed = derive_seed(seed, i + 1);
        TrialResult r = c.kind == "chase" ? run_chase_trial(ctx, arena, words, cfg, c, tseed)
                                          : run_bw_trial(ctx, arena, words, cfg, c, tseed);
        if (!r.valid) ++invalid;

        // Derived quantities (docs/measurement.md §2). Computed for every
        // row, including invalid ones; invalid rows are flagged, never dropped.
        //   primary:      single long encoder, time per iteration
        //   short:        single short encoder (same trial; NOT independent)
        //   differential: (long - short) / (n_long - n_short), bias check
        const double dT = r.gpu_long - r.gpu_short;
        double lat = NAN, lat_s = NAN, lat_d = NAN, bw = NAN, bw_s = NAN, bw_d = NAN;
        if (c.kind == "bandwidth") {
            bw = r.bytes_long / r.gpu_long;  // bytes per ns == GB/s (10^9 B/s)
            bw_s = r.bytes_short / r.gpu_short;
            bw_d = (r.bytes_long - r.bytes_short) / dT;
        } else {
            lat = r.gpu_long / static_cast<double>(c.n_long);
            lat_s = r.gpu_short / static_cast<double>(c.n_short);
            lat_d = dT / static_cast<double>(c.n_long - c.n_short);
        }

        csv.row({session_id + "-" + std::to_string(i), session_id, cfg.phase, fmt_u(i), utc_now_iso(),
                 c.experiment, c.kind, c.pattern, c.sentinel ? "1" : "0", std::to_string(tr.rep),
                 fmt_u(c.ws_bytes), fmt_u(c.stride_bytes), fmt_u(c.slots), fmt_u(r.threads), fmt_u(r.tg),
                 fmt_u(c.n_long), fmt_u(c.n_short), fmt_u(c.n_warm), r.long_first ? "1" : "0",
                 fmt_f(r.gpu_short), fmt_f(r.gpu_long), fmt_f(r.gpu_warm), fmt_f(r.alu0), fmt_f(r.alu1), fmt_f(r.alu2),
                 fmt_f(r.seq.cmdbuf_ns), fmt_f(r.seq.cpu_wall_ns),
                 r.seq.ok ? fmt_u(r.seq.tick_start[1]) : "", fmt_f(gap(r.seq, 3)), fmt_f(gap(r.seq, 5)),
                 fmt_f(max_gap(r.seq)), fmt_f(r.gpu_long),
                 fmt_f(lat), fmt_f(lat_s), fmt_f(lat_d), fmt_f(r.bytes_short), fmt_f(r.bytes_long),
                 fmt_f(bw), fmt_f(bw_s), fmt_f(bw_d),
                 r.valid ? "1" : "0", r.note, thermal_state_name(), std::to_string(tseed)});

        if ((i + 1) % 25 == 0 || i + 1 == schedule.size()) {
            std::printf("  [%zu/%zu] invalid=%zu thermal=%s\n", i + 1, schedule.size(), invalid, thermal_state_name().c_str());
            std::fflush(stdout);
            write_meta("running", i + 1, invalid, nullptr);
        }
    }
    const ClockCalibration cal1 = ctx.calibrate(0.2);
    write_meta("completed", schedule.size(), invalid, &cal1);
    std::printf("wrote %s\n      %s\n", csv_path.c_str(), json_path.c_str());
    return invalid == 0 ? 0 : 2;
}

// =========================================================================
// validate — measurement-validation protocol v3 (docs/measurement.md §3)
//
// Records individual timed encoders (not differential trials). Every command
// buffer below is additionally preceded (inside the same command buffer) by
// the same clock spin-up encoder used in benchmark trials.
//
//   mode=linearity     cb[ALU ref][warm-up chase][timed chase n], n on a
//                      geometric grid: tests T(n) = a + b·n, the size of a,
//                      the transient below one traversal, and invariance of b
//                      to the random permutation (chain seed).
//   mode=alu           cb[ALU warm-up][timed ALU n]: linearity of the clock
//                      reference itself.
//   mode=persist_same_cb      cb[ALU][warm-up chase][timed chase, n = slots]
//   mode=persist_separate_cb  cb1[ALU][warm-up chase]; cb2[ALU][timed chase, n = slots]
//                      compares one traversal right after a warm-up in the
//                      same vs. a separate command buffer: tests whether GPU
//                      cache contents survive a command-buffer boundary.
// =========================================================================
static int cmd_validate(const Args& args) {
    MetalCtx ctx;
    ClockCalibration cal0;
    if (!init_ctx(ctx, cal0)) return 1;
    std::string err;
    const std::vector<uint64_t> ws_list = {4ull << 10, 256ull << 10, 16ull << 20, 128ull << 20};
    const uint64_t stride = 64, alu_ref_n = 1u << 16;
    const int n_seeds = 3, reps = 3, persist_reps = 5;
    std::vector<uint64_t> n_list = {0};
    for (int e = 10; e <= 22; ++e) n_list.push_back(1ull << e);

    id<MTLBuffer> arena = alloc_arena(ctx, ws_list.back(), err);
    if (!arena) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    uint32_t* words = static_cast<uint32_t*>(arena.contents);

    const SystemInfo si = collect_system_info();
    warn_conditions(si);
    const uint64_t seed = args.has_seed ? args.seed : entropy_seed();
    const std::string session_id = args.session_id.empty() ? utc_now_compact() : args.session_id;
    const std::string out_dir = args.out.empty() ? std::string(SA_SOURCE_DIR) + "/data/validation" : args.out;
    make_dirs(out_dir);
    const std::string csv_path = out_dir + "/" + session_id + ".csv";
    CsvWriter csv;
    if (!csv.open(csv_path, {"session_id", "launch_index", "timestamp_utc", "block", "mode", "working_set_bytes",
                             "stride_bytes", "n_slots", "chain_seed_index", "iteration_count", "warmup_iterations",
                             "gpu_ns", "gpu_ns_warmup", "alu_ns_per_op", "cmdbuf_ns", "cpu_wall_ns",
                             "validation_ok", "thermal_state"})) {
        std::fprintf(stderr, "error: cannot create %s\n", csv_path.c_str());
        return 1;
    }

    struct Block { std::string mode; uint64_t ws; int seed_idx; };
    std::vector<Block> blocks;
    for (uint64_t ws : ws_list) {
        for (int s = 0; s < n_seeds; ++s) blocks.push_back({"linearity", ws, s});
        blocks.push_back({"persist", ws, 0});
    }
    for (int s = 0; s < n_seeds; ++s) blocks.push_back({"alu", 0, s});
    Rng order(derive_seed(seed, 0));
    for (size_t i = blocks.size(); i > 1; --i) std::swap(blocks[i - 1], blocks[order.below(i)]);

    // Prepend the clock spin-up encoder (result slot 7) to a sequence and drop
    // its timing from the result, so indices refer to the caller's encoders.
    const uint64_t spin_n = 1ull << 21;
    const uint32_t spin_ok = cpu_lcg(1, spin_n);
    auto run_spun = [&](std::vector<Dispatch> seq) {
        ctx.out_word(7)[0] = kUnwritten;
        seq.insert(seq.begin(), alu_dispatch(ctx, 7, 1, spin_n));
        SequenceTiming t = ctx.run(seq);
        if (t.ok && ctx.out_word(7)[0] != spin_ok) { t.ok = false; t.error = "spin-up mismatch"; }
        if (t.ok) {
            t.gpu_ns.erase(t.gpu_ns.begin());
            t.tick_start.erase(t.tick_start.begin());
            t.tick_end.erase(t.tick_end.begin());
        }
        return t;
    };

    session_warmup(ctx, 2.0);
    size_t li = 0, invalid = 0;
    auto emit = [&](size_t bi, const std::string& mode, const Block& b, uint64_t slots, uint64_t n, uint64_t n_warm,
                    double gpu, double gpu_warm, double alu, double cmdbuf, double wall, bool ok) {
        if (!ok) ++invalid;
        const bool chase = b.mode != "alu";
        csv.row({session_id, fmt_u(li++), utc_now_iso(), fmt_u(bi), mode, chase ? fmt_u(b.ws) : "",
                 chase ? fmt_u(stride) : "", chase ? fmt_u(slots) : "", std::to_string(b.seed_idx), fmt_u(n),
                 fmt_u(n_warm), fmt_f(gpu), fmt_f(gpu_warm), fmt_f(alu), fmt_f(cmdbuf), fmt_f(wall), ok ? "1" : "0",
                 thermal_state_name()});
    };

    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        const Block& b = blocks[bi];
        Rng rng(derive_seed(seed, 1000 + bi));
        const uint64_t slots = b.ws / stride;
        const uint64_t n_warm = std::max<uint64_t>(slots, 1u << 16);
        if (b.mode != "alu") build_chain(words, slots, stride / 4, Pattern::Random, rng);
        const uint32_t alu_seed = static_cast<uint32_t>(rng.next()) | 1u;
        const uint32_t alu_ok = cpu_lcg(alu_seed, alu_ref_n);

        if (b.mode == "linearity" || b.mode == "alu") {
            std::vector<uint64_t> ns;
            for (int r = 0; r < reps; ++r) ns.insert(ns.end(), n_list.begin(), n_list.end());
            for (size_t i = ns.size(); i > 1; --i) std::swap(ns[i - 1], ns[rng.below(i)]);
            for (uint64_t n : ns) {
                ctx.out_word(0)[0] = ctx.out_word(1)[0] = ctx.out_word(2)[0] = kUnwritten;
                if (b.mode == "linearity") {
                    SequenceTiming t = run_spun({alu_dispatch(ctx, 0, alu_seed, alu_ref_n),
                                                chase_dispatch(ctx, arena, 1, 0, n_warm),
                                                chase_dispatch(ctx, arena, 2, 0, n)});
                    const bool ok = t.ok && ctx.out_word(0)[0] == alu_ok &&
                                    ctx.out_word(1)[0] == cpu_chase(words, 0, n_warm) &&
                                    ctx.out_word(2)[0] == cpu_chase(words, 0, n);
                    emit(bi, b.mode, b, slots, n, n_warm, t.ok ? t.gpu_ns[2] : NAN, t.ok ? t.gpu_ns[1] : NAN,
                         t.ok ? t.gpu_ns[0] / alu_ref_n : NAN, t.cmdbuf_ns, t.cpu_wall_ns, ok);
                } else {
                    SequenceTiming t = run_spun({alu_dispatch(ctx, 0, alu_seed, alu_ref_n),
                                                alu_dispatch(ctx, 1, alu_seed, n)});
                    const bool ok = t.ok && ctx.out_word(0)[0] == alu_ok && ctx.out_word(1)[0] == cpu_lcg(alu_seed, n);
                    emit(bi, b.mode, b, 0, n, alu_ref_n, t.ok ? t.gpu_ns[1] : NAN, t.ok ? t.gpu_ns[0] : NAN,
                         t.ok ? t.gpu_ns[0] / alu_ref_n : NAN, t.cmdbuf_ns, t.cpu_wall_ns, ok);
                }
            }
        } else {  // persist
            const uint32_t end_one = cpu_chase(words, 0, slots);
            const uint32_t end_warm = cpu_chase(words, 0, n_warm);
            for (int r = 0; r < persist_reps; ++r) {
                const bool same_first = rng.below(2) == 1;
                for (int k = 0; k < 2; ++k) {
                    const bool same = (k == 0) == same_first;
                    for (uint32_t s = 0; s < 3; ++s) ctx.out_word(s)[0] = kUnwritten;
                    if (same) {
                        SequenceTiming t = run_spun({alu_dispatch(ctx, 0, alu_seed, alu_ref_n),
                                                    chase_dispatch(ctx, arena, 1, 0, n_warm),
                                                    chase_dispatch(ctx, arena, 2, 0, slots)});
                        const bool ok = t.ok && ctx.out_word(0)[0] == alu_ok && ctx.out_word(1)[0] == end_warm &&
                                        ctx.out_word(2)[0] == end_one;
                        emit(bi, "persist_same_cb", b, slots, slots, n_warm, t.ok ? t.gpu_ns[2] : NAN,
                             t.ok ? t.gpu_ns[1] : NAN, t.ok ? t.gpu_ns[0] / alu_ref_n : NAN, t.cmdbuf_ns,
                             t.cpu_wall_ns, ok);
                    } else {
                        SequenceTiming t1 = run_spun({alu_dispatch(ctx, 0, alu_seed, alu_ref_n),
                                                     chase_dispatch(ctx, arena, 1, 0, n_warm)});
                        const bool ok1 = t1.ok && ctx.out_word(0)[0] == alu_ok && ctx.out_word(1)[0] == end_warm;
                        ctx.out_word(0)[0] = ctx.out_word(2)[0] = kUnwritten;
                        SequenceTiming t2 = run_spun({alu_dispatch(ctx, 0, alu_seed, alu_ref_n),
                                                     chase_dispatch(ctx, arena, 2, 0, slots)});
                        const bool ok = ok1 && t2.ok && ctx.out_word(0)[0] == alu_ok && ctx.out_word(2)[0] == end_one;
                        emit(bi, "persist_separate_cb", b, slots, slots, n_warm, t2.ok ? t2.gpu_ns[1] : NAN,
                             t1.ok ? t1.gpu_ns[1] : NAN, t2.ok ? t2.gpu_ns[0] / alu_ref_n : NAN, t2.cmdbuf_ns,
                             t2.cpu_wall_ns, ok);
                    }
                }
            }
        }
        std::printf("  block %zu/%zu (%s ws=%llu seed=%d) done\n", bi + 1, blocks.size(), b.mode.c_str(),
                    static_cast<unsigned long long>(b.ws), b.seed_idx);
        std::fflush(stdout);
    }
    const ClockCalibration cal1 = ctx.calibrate(0.2);
    Json j;
    j.str("session_id", session_id).str("protocol", "validate-v3").str("session_seed", std::to_string(seed))
     .integer("launches", static_cast<long long>(li)).integer("invalid", static_cast<long long>(invalid))
     .raw("clock_calibration_start", calib_json(cal0)).raw("clock_calibration_end", calib_json(cal1))
     .raw("device", device_json(ctx)).raw("system", system_info_json(si));
    write_text_file(out_dir + "/" + session_id + ".json", j.dump());
    std::printf("wrote %s (%zu rows, %zu invalid)\n", csv_path.c_str(), li, invalid);
    return invalid == 0 ? 0 : 2;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        Args args;
        if (!parse_args(argc, argv, args)) { usage(); return 1; }
        if (args.cmd == "info") return cmd_info();
        if (args.cmd == "benchmark") return cmd_benchmark(args);
        if (args.cmd == "validate") return cmd_validate(args);
        usage();
        return 1;
    }
}
