// rng.h — small, fully specified pseudo-random number generator.
//
// We deliberately do NOT use std::mt19937 + std::uniform_int_distribution:
// the distribution algorithms are implementation-defined, so the same seed can
// produce different sequences on different standard libraries. SplitMix64
// with explicit rejection sampling is defined bit-for-bit by this file, which
// makes every recorded seed exactly reproducible.
#pragma once
#include <cstdint>

struct Rng {
    uint64_t state;

    explicit Rng(uint64_t seed) : state(seed) {}

    // SplitMix64 (Steele, Lea & Flood 2014; public-domain reference by Vigna).
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform integer in [0, n), n > 0, without modulo bias.
    // Values below `threshold` = 2^64 mod n are rejected so that the accepted
    // range has a length that is an exact multiple of n.
    uint64_t below(uint64_t n) {
        const uint64_t threshold = (0 - n) % n;
        for (;;) {
            const uint64_t r = next();
            if (r >= threshold) return r % n;
        }
    }
};

// Derive an independent child seed from a parent seed and an index
// (one SplitMix64 step on a mixed input).
inline uint64_t derive_seed(uint64_t parent, uint64_t index) {
    Rng r(parent ^ (index * 0xD1B54A32D192ED03ull));
    return r.next();
}
