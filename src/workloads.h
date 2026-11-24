// workloads.h — construction of GPU input buffers and their CPU reference results.
#pragma once
#include <cstdint>
#include "rng.h"

enum class Pattern { Random, Sequential };

// Build a pointer chain inside `words` (32-bit words).
//
// The working set is divided into `slots` slots placed `stride_words` words
// apart; only the first word of each slot is used. words[k*stride_words]
// holds the word index of the next slot.
//
//   Random:     a uniformly random single cycle through all slots
//               (Sattolo's algorithm). Every slot is visited exactly once per
//               traversal, and the visiting order defeats address prediction.
//   Sequential: slot k -> slot k+1 (mod slots), i.e. a constant stride.
void build_chain(uint32_t* words, uint64_t slots, uint64_t stride_words,
                 Pattern pattern, Rng& rng);

// CPU reference of the `chase` kernel: follow n links from `start`.
uint32_t cpu_chase(const uint32_t* words, uint32_t start, uint64_t n);

// Number of links followed from slot 0 until slot 0 is reached again
// (returns 0 if not reached within max_steps). Used by tests to prove the
// chain is a single cycle of length `slots`.
uint64_t cycle_length(const uint32_t* words, uint64_t max_steps);

// CPU reference of the `bw_read` kernel: the wrapping 32-bit total of all
// per-thread accumulators, for a grid of `threads` threads.
uint32_t cpu_bw_reference(const uint32_t* words, uint64_t n_vec,
                          uint32_t passes, uint32_t threads);

// CPU reference of the `alu_chain` kernel.
uint32_t cpu_lcg(uint32_t seed, uint64_t n);

// Fill `n_words` words with pseudo-random content.
void fill_random(uint32_t* words, uint64_t n_words, Rng& rng);
