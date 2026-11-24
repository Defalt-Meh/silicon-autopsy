// memory.metal — GPU kernels for SILICON AUTOPSY.
//
// Design rules applied to every kernel:
//   1. Every result the kernel computes is written to device memory, so the
//      compiler cannot prove the work is dead and remove it.
//   2. Every result is reproducible on the CPU (see src/workloads.cpp), so each
//      launch can be validated bit-exactly.
//   3. Integer arithmetic only: no fast-math reassociation can change results.

#include <metal_stdlib>
#include "shared_types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// chase — dependent pointer chasing (latency probe).
//
// buf holds a chain of 32-bit word indices: buf[i] is the index of the next
// element. Each load's address depends on the value returned by the previous
// load, so loads cannot overlap or be issued speculatively; the elapsed time
// per step is therefore the full effective load-to-use latency (plus the few
// ALU instructions of the loop, which are a constant per step).
//
// Launched as a single thread (grid = 1): we measure latency, not throughput.
// ---------------------------------------------------------------------------
kernel void chase(device const uint*      buf [[buffer(0)]],
                  device uint*            out [[buffer(1)]],
                  constant ChaseArgs&     a   [[buffer(2)]])
{
    uint i = a.start;
    for (uint k = 0; k < a.n; ++k) {
        i = buf[i];
    }
    out[0] = i;  // final index: consumed by the host validator
}

// ---------------------------------------------------------------------------
// bw_read — streaming read bandwidth probe.
//
// Thread g reads vectors g, g+G, g+2G, ... (G = grid size), so on every
// iteration consecutive threads read consecutive 16-byte vectors (coalesced).
// The whole buffer is read `passes` times.
//
// The accumulator uses a rotate before every add. This makes the per-thread
// result depend on the ORDER of the loads, so the compiler cannot collapse
// the `passes` loop into "passes × (sum of one pass)" and skip memory reads.
// Loads do not depend on `acc`, so they can still be issued far ahead.
//
// Per-thread results are reduced in the SIMD group and added atomically to
// out[0] (wrapping 32-bit arithmetic, order independent).
// ---------------------------------------------------------------------------
kernel void bw_read(device const uint4*   buf   [[buffer(0)]],
                    device atomic_uint*   out   [[buffer(1)]],
                    constant BwArgs&      a     [[buffer(2)]],
                    uint                  gid   [[thread_position_in_grid]],
                    uint                  gsize [[threads_per_grid]])
{
    uint acc = 0;
    for (uint p = 0; p < a.passes; ++p) {
        for (uint k = gid; k < a.n_vec; k += gsize) {
            uint4 v = buf[k];
            acc = rotate(acc, 1u) + (v.x ^ v.y) + (v.z ^ v.w);
        }
    }
    acc = simd_sum(acc);
    if (simd_is_first()) {
        atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// alu_chain — clock-frequency sentinel.
//
// A chain of n dependent integer multiply-adds (a linear congruential
// generator). Its duration depends only on the GPU core clock and the fixed
// latency of the multiply-add, not on memory. Measured periodically during
// a session, it tracks GPU frequency drift without any manufacturer data.
// ---------------------------------------------------------------------------
kernel void alu_chain(device uint*        out [[buffer(1)]],
                      constant AluArgs&   a   [[buffer(2)]])
{
    uint x = a.seed;
    for (uint k = 0; k < a.n; ++k) {
        x = x * 1664525u + 1013904223u;
    }
    out[0] = x;
}
