// test_engine.cpp — unit tests for the GPU-independent parts of the engine:
// RNG, chain construction, CPU reference kernels, config parsing, scheduling.
// No framework dependency: each CHECK prints a message and counts failures.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "config.h"
#include "rng.h"
#include "workloads.h"

static int failures = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) { ++failures; std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
    } while (0)

static void test_rng_uniformity() {
    // Chi-square check of Rng::below(10) over 100k draws (df = 9).
    // P(chi2_9 > 27.88) = 0.001, so a correct generator fails ~0.1% of seeds;
    // the seed is fixed, so this test is deterministic.
    Rng r(12345);
    std::vector<int> counts(10, 0);
    const int n = 100000;
    for (int i = 0; i < n; ++i) counts[r.below(10)]++;
    double chi2 = 0;
    for (int c : counts) chi2 += (c - n / 10.0) * (c - n / 10.0) / (n / 10.0);
    CHECK(chi2 < 27.88, "Rng::below chi-square uniformity");
    // Determinism: identical seeds give identical streams.
    Rng a(7), b(7);
    bool same = true;
    for (int i = 0; i < 100; ++i) same &= a.next() == b.next();
    CHECK(same, "Rng determinism");
}

static void test_sattolo_single_cycle() {
    for (uint64_t slots : {2ull, 3ull, 17ull, 1000ull, 65536ull}) {
        for (uint64_t sw : {1ull, 16ull}) {
            std::vector<uint32_t> w(slots * sw, 0xDEADBEEF);
            Rng r(slots * 31 + sw);
            build_chain(w.data(), slots, sw, Pattern::Random, r);
            CHECK(cycle_length(w.data(), slots + 1) == slots, "random chain is a single cycle over all slots");
            // Every link points to a slot start.
            bool aligned = true;
            for (uint64_t k = 0; k < slots; ++k) aligned &= (w[k * sw] % sw) == 0 && w[k * sw] / sw < slots;
            CHECK(aligned, "random chain links point to slot starts");
        }
    }
    std::vector<uint32_t> s(64 * 4);
    Rng r(1);
    build_chain(s.data(), 64, 4, Pattern::Sequential, r);
    CHECK(cycle_length(s.data(), 65) == 64, "sequential chain is a single cycle");
    CHECK(s[0] == 4 && s[63 * 4] == 0, "sequential chain links are +stride mod W");
    CHECK(cpu_chase(s.data(), 0, 64) == 0 && cpu_chase(s.data(), 0, 3) == 12, "cpu_chase follows links");
}

static void test_sattolo_uniform_over_cycles() {
    // For 4 slots there are (4-1)! = 6 cyclic permutations; Sattolo must hit
    // each with probability 1/6. Chi-square with df = 5, critical 20.52 at 0.001.
    std::map<std::vector<uint32_t>, int> counts;
    Rng r(99);
    const int n = 60000;
    for (int i = 0; i < n; ++i) {
        std::vector<uint32_t> w(4);
        build_chain(w.data(), 4, 1, Pattern::Random, r);
        counts[w]++;
    }
    CHECK(counts.size() == 6, "Sattolo produces exactly the 6 cyclic permutations of 4 elements");
    double chi2 = 0;
    for (auto& kv : counts) chi2 += (kv.second - n / 6.0) * (kv.second - n / 6.0) / (n / 6.0);
    CHECK(chi2 < 20.52, "Sattolo uniform over cyclic permutations");
}

static void test_bw_reference() {
    // Thread-major emulation (literal transcription of the kernel) must equal
    // the k-major implementation used by the engine.
    const uint64_t n_vec = 1000;
    std::vector<uint32_t> w(n_vec * 4);
    Rng r(5);
    fill_random(w.data(), w.size(), r);
    for (uint32_t threads : {1u, 7u, 64u, 4096u}) {
        for (uint32_t passes : {1u, 3u}) {
            uint32_t total = 0;
            for (uint32_t g = 0; g < threads; ++g) {
                uint32_t acc = 0;
                for (uint32_t p = 0; p < passes; ++p)
                    for (uint64_t k = g; k < n_vec; k += threads) {
                        const uint32_t* v = &w[4 * k];
                        acc = ((acc << 1) | (acc >> 31)) + (v[0] ^ v[1]) + (v[2] ^ v[3]);
                    }
                total += acc;
            }
            CHECK(total == cpu_bw_reference(w.data(), n_vec, passes, threads), "bandwidth reference matches kernel semantics");
        }
    }
    // The rotate makes the result order dependent: 2 passes != 2 x (1 pass) in general.
    CHECK(cpu_bw_reference(w.data(), n_vec, 2, 1) != 2u * cpu_bw_reference(w.data(), n_vec, 1, 1),
          "bandwidth accumulator is not pass-separable");
}

static void test_config() {
    uint64_t v = 0;
    CHECK(parse_size("1K", v) && v == 1024, "parse 1K");
    CHECK(parse_size("128MiB", v) && v == 128ull << 20, "parse 128MiB");
    CHECK(parse_size("64", v) && v == 64, "parse 64");
    CHECK(!parse_size("12Q", v), "reject bad suffix");

    std::vector<uint64_t> ls;
    std::string err;
    CHECK(parse_size_list("logspace 1K 8K 2", ls, err) && ls.size() == 7, "logspace point count");
    CHECK(ls.front() == 1024 && ls.back() == 8192 && ls[1] == 1448, "logspace values");

    const std::string text =
        "name = t\nphase = smoke\nreps = 3\nsentinel_every = 4\n"
        "[a]\nkind = chase\npattern = random\nws = 1K, 4K, 100\nstride = 64, 2K\n"
        "[b]\nkind = bandwidth\nws = 1M\n";
    Config cfg;
    CHECK(parse_config(text, cfg, err), ("parse_config: " + err).c_str());
    std::vector<std::string> notes;
    const auto cells = expand_cells(cfg, notes);
    // stride 64: 1K, 4K, 64 (100 rounded down to 1 slot -> skipped) -> 2 cells
    // stride 2K: 1K skipped, 4K ok, 100 skipped -> 1 cell; bandwidth 1 cell
    CHECK(cells.size() == 4, "expand_cells count");
    CHECK(notes.size() == 3, "expand_cells skip notes");
    for (const auto& c : cells) CHECK(c.ws_bytes == c.slots * c.stride_bytes, "ws = slots * stride");

    const auto sched = make_schedule(cfg, cells, 42);
    int non_sentinel = 0, sentinels = 0;
    for (const auto& t : sched) (t.cell.sentinel ? sentinels : non_sentinel)++;
    CHECK(non_sentinel == 12, "schedule contains every cell x reps");
    CHECK(sentinels == 4, "sentinel at start and after every 4 trials (12/4 = 3 + initial)");
    for (const auto& c : cells) {
        if (c.kind == "chase") {
            CHECK(c.n_short >= c.slots && c.n_short >= cfg.n_min, "n_short covers a full traversal");
            CHECK(c.n_short >= std::min<uint64_t>(8 * c.slots, cfg.n_budget), "n_short reaches target traversals within budget");
            CHECK(c.n_warm >= c.slots, "warm-up covers a full traversal");
            CHECK(c.n_long == cfg.long_ratio * c.n_short && c.n_long <= cfg.n_max, "n_long = long_ratio * n_short <= n_max");
        }
    }
    // A cell that cannot complete a traversal within n_max must be skipped.
    Config tight;
    CHECK(parse_config("n_min = 1K\nn_max = 4K\ntarget_traversals = 1\n[a]\nkind = chase\npattern = random\nws = 64K, 1M\nstride = 64\n", tight, err), "parse tight");
    std::vector<std::string> tnotes;
    const auto tcells = expand_cells(tight, tnotes);
    CHECK(tcells.size() == 1 && tcells[0].ws_bytes == (64u << 10), "n_max violation skips the cell");
    const auto sched2 = make_schedule(cfg, cells, 42);
    bool same = sched.size() == sched2.size();
    for (size_t i = 0; same && i < sched.size(); ++i)
        same = sched[i].cell.ws_bytes == sched2[i].cell.ws_bytes && sched[i].cell.experiment == sched2[i].cell.experiment;
    CHECK(same, "schedule reproducible from seed");

    Config bad;
    CHECK(!parse_config("[x]\nkind = chase\npattern = random\nws = 1K\nstride = 6\n", bad, err), "reject stride not multiple of 4");
}

int main() {
    test_rng_uniformity();
    test_sattolo_single_cycle();
    test_sattolo_uniform_over_cycles();
    test_bw_reference();
    test_config();
    if (failures == 0) std::printf("all engine tests passed\n");
    return failures == 0 ? 0 : 1;
}
