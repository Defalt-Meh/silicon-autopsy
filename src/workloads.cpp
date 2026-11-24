// workloads.cpp — see workloads.h.
#include "workloads.h"

#include <utility>
#include <vector>

void build_chain(uint32_t* words, uint64_t slots, uint64_t stride_words,
                 Pattern pattern, Rng& rng) {
    if (pattern == Pattern::Sequential) {
        for (uint64_t k = 0; k < slots; ++k) {
            words[k * stride_words] = static_cast<uint32_t>(((k + 1) % slots) * stride_words);
        }
        return;
    }
    // Sattolo's algorithm, performed in place on the slot array.
    // Start from the identity a[k] = k; for k = slots-1 down to 1, swap a[k]
    // with a[j], j uniform in [0, k-1] (strictly below k). The result, read as
    // the map k -> a[k], is a uniformly random cyclic permutation: one cycle
    // containing every slot (Sattolo 1986).
    for (uint64_t k = 0; k < slots; ++k) {
        words[k * stride_words] = static_cast<uint32_t>(k * stride_words);
    }
    for (uint64_t k = slots - 1; k > 0; --k) {
        const uint64_t j = rng.below(k);
        std::swap(words[k * stride_words], words[j * stride_words]);
    }
}

uint32_t cpu_chase(const uint32_t* words, uint32_t start, uint64_t n) {
    uint32_t i = start;
    for (uint64_t k = 0; k < n; ++k) i = words[i];
    return i;
}

uint64_t cycle_length(const uint32_t* words, uint64_t max_steps) {
    uint32_t i = words[0];
    for (uint64_t k = 1; k <= max_steps; ++k) {
        if (i == 0) return k;
        i = words[i];
    }
    return 0;
}

static inline uint32_t rotl1(uint32_t x) { return (x << 1) | (x >> 31); }

uint32_t cpu_bw_reference(const uint32_t* words, uint64_t n_vec,
                          uint32_t passes, uint32_t threads) {
    // Thread g processes vectors g, g+T, g+2T, ... in increasing order, once
    // per pass. Iterating k sequentially and updating acc[k % T] preserves
    // that per-thread order while reading memory sequentially (fast on CPU).
    std::vector<uint32_t> acc(threads, 0);
    for (uint32_t p = 0; p < passes; ++p) {
        for (uint64_t k = 0; k < n_vec; ++k) {
            const uint32_t* v = words + 4 * k;
            uint32_t& a = acc[k % threads];
            a = rotl1(a) + (v[0] ^ v[1]) + (v[2] ^ v[3]);
        }
    }
    uint32_t total = 0;
    for (uint32_t a : acc) total += a;  // wrapping, order independent
    return total;
}

uint32_t cpu_lcg(uint32_t seed, uint64_t n) {
    uint32_t x = seed;
    for (uint64_t k = 0; k < n; ++k) x = x * 1664525u + 1013904223u;
    return x;
}

void fill_random(uint32_t* words, uint64_t n_words, Rng& rng) {
    uint64_t k = 0;
    for (; k + 1 < n_words; k += 2) {
        const uint64_t r = rng.next();
        words[k] = static_cast<uint32_t>(r);
        words[k + 1] = static_cast<uint32_t>(r >> 32);
    }
    if (k < n_words) words[k] = static_cast<uint32_t>(rng.next());
}
