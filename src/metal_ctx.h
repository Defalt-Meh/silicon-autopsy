// metal_ctx.h — Metal device, pipelines and timing of encoder sequences.
//
// Every trial is ONE command buffer containing several compute encoders, one
// dispatch per encoder. Validation experiments (docs/measurement.md §3)
// showed that GPU cache contents do not survive a command-buffer boundary on
// this device, so warm-up and timed launches must share a command buffer.
//
// Clocks recorded:
//   gpu_ns[k]   GPU timestamp counters sampled at the start and end of encoder
//               k (MTLCounterSamplingPointAtStageBoundary — the only sampling
//               point this GPU supports), converted to ns with a per-session
//               calibrated tick period.
//   cmdbuf_ns   MTLCommandBuffer GPUEndTime − GPUStartTime (whole sequence).
//   cpu_wall_ns CPU monotonic clock from just before -commit to the return of
//               -waitUntilCompleted (whole sequence, incl. submission and
//               completion-notification overhead).
#pragma once
#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <vector>

// One dispatch = one encoder. Kernel arguments: buffer 0 = `b0` (may be nil),
// buffer 1 = the shared result buffer at `out_slot` × 16 bytes, buffer 2 = args.
struct Dispatch {
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer> b0 = nil;
    uint32_t out_slot = 0;
    std::vector<uint8_t> args;
    MTLSize grid = MTLSizeMake(1, 1, 1);
    MTLSize tg = MTLSizeMake(1, 1, 1);
};

struct SequenceTiming {
    bool ok = false;
    std::string error;
    std::vector<uint64_t> tick_start, tick_end;  // per encoder
    std::vector<double> gpu_ns;                  // per encoder
    double ns_per_tick = 0;                      // conversion used for gpu_ns
    double cmdbuf_ns = 0;
    double cpu_wall_ns = 0;
};

struct ClockCalibration {
    double ns_per_tick = 0;  // CPU-uptime nanoseconds per GPU timestamp tick
    double interval_s = 0;
    uint64_t gpu_ticks = 0;
    uint64_t uptime_ns = 0;
};

class MetalCtx {
public:
    static constexpr uint32_t kMaxEncoders = 8;

    bool init(const std::string& metallib_path, std::string& err);

    // Encode all dispatches into a single command buffer (one encoder each),
    // commit, wait for completion and return per-encoder GPU timings.
    SequenceTiming run(const std::vector<Dispatch>& seq);

    // Measure the GPU timestamp tick period against CLOCK_UPTIME_RAW over
    // `seconds`, instead of assuming a nominal tick frequency.
    ClockCalibration calibrate(double seconds);

    // Result words: out_word(slot)[0] is what the kernel at `slot` wrote.
    uint32_t* out_word(uint32_t slot) { return static_cast<uint32_t*>(out.contents) + 4 * slot; }

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso_chase = nil, pso_bw = nil, pso_alu = nil;
    id<MTLBuffer> out = nil;  // kMaxEncoders × 16 bytes
    double ns_per_tick = 1.0;

private:
    id<MTLCounterSampleBuffer> samples_ = nil;
};

template <typename T>
std::vector<uint8_t> as_bytes(const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    return std::vector<uint8_t>(p, p + sizeof(T));
}

std::string thermal_state_name();
