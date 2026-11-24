// metal_ctx.mm — see metal_ctx.h.
#import "metal_ctx.h"

#include <time.h>
#include <unistd.h>

static uint64_t uptime_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                            NSString* name, std::string& err) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { err = "kernel not found: " + std::string(name.UTF8String); return nil; }
    NSError* e = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!pso) err = "pipeline creation failed: " + std::string(e.localizedDescription.UTF8String);
    return pso;
}

bool MetalCtx::init(const std::string& metallib_path, std::string& err) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) { err = "no Metal device"; return false; }
    queue = [device newCommandQueue];

    NSError* e = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path.c_str()]];
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&e];
    if (!lib) { err = "cannot load " + metallib_path + ": " + std::string(e.localizedDescription.UTF8String); return false; }

    if (!(pso_chase = make_pso(device, lib, @"chase", err))) return false;
    if (!(pso_bw = make_pso(device, lib, @"bw_read", err))) return false;
    if (!(pso_alu = make_pso(device, lib, @"alu_chain", err))) return false;

    out = [device newBufferWithLength:16 * kMaxEncoders options:MTLResourceStorageModeShared];

    if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        err = "device does not support stage-boundary counter sampling";
        return false;
    }
    id<MTLCounterSet> ts = nil;
    for (id<MTLCounterSet> cs in device.counterSets)
        if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = cs;
    if (!ts) { err = "no timestamp counter set"; return false; }
    MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = ts;
    d.storageMode = MTLStorageModeShared;
    d.sampleCount = 2 * kMaxEncoders;
    samples_ = [device newCounterSampleBufferWithDescriptor:d error:&e];
    if (!samples_) { err = "cannot create counter sample buffer"; return false; }
    return true;
}

SequenceTiming MetalCtx::run(const std::vector<Dispatch>& seq) {
    SequenceTiming t;
    t.ns_per_tick = ns_per_tick;
    if (seq.empty() || seq.size() > kMaxEncoders) { t.error = "bad sequence length"; return t; }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (size_t k = 0; k < seq.size(); ++k) {
            const Dispatch& d = seq[k];
            MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
            pd.dispatchType = MTLDispatchTypeSerial;
            pd.sampleBufferAttachments[0].sampleBuffer = samples_;
            pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * k;
            pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * k + 1;
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:pd];
            [enc setComputePipelineState:d.pso];
            if (d.b0) [enc setBuffer:d.b0 offset:0 atIndex:0];
            [enc setBuffer:out offset:16 * d.out_slot atIndex:1];
            [enc setBytes:d.args.data() length:d.args.size() atIndex:2];
            [enc dispatchThreads:d.grid threadsPerThreadgroup:d.tg];
            [enc endEncoding];
        }

        // Encoding is finished before the CPU clock starts.
        const uint64_t c0 = uptime_ns();
        [cb commit];
        [cb waitUntilCompleted];
        const uint64_t c1 = uptime_ns();
        t.cpu_wall_ns = static_cast<double>(c1 - c0);

        if (cb.status != MTLCommandBufferStatusCompleted) {
            t.error = cb.error ? cb.error.localizedDescription.UTF8String : "command buffer not completed";
            return t;
        }
        t.cmdbuf_ns = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;

        const NSUInteger ns = 2 * seq.size();
        NSData* data = [samples_ resolveCounterRange:NSMakeRange(0, ns)];
        if (!data || data.length < ns * sizeof(MTLCounterResultTimestamp)) { t.error = "counter resolve failed"; return t; }
        const auto* r = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
        for (size_t k = 0; k < seq.size(); ++k) {
            const uint64_t a = r[2 * k].timestamp, b = r[2 * k + 1].timestamp;
            if (a == MTLCounterErrorValue || b == MTLCounterErrorValue || b < a) {
                t.error = "invalid GPU timestamp sample";
                return t;
            }
            t.tick_start.push_back(a);
            t.tick_end.push_back(b);
            t.gpu_ns.push_back(static_cast<double>(b - a) * ns_per_tick);
        }
        t.ok = true;
    }
    return t;
}

ClockCalibration MetalCtx::calibrate(double seconds) {
    MTLTimestamp c0, g0, c1, g1;
    const uint64_t u0 = uptime_ns();
    [device sampleTimestamps:&c0 gpuTimestamp:&g0];
    usleep(static_cast<useconds_t>(seconds * 1e6));
    const uint64_t u1 = uptime_ns();
    [device sampleTimestamps:&c1 gpuTimestamp:&g1];
    ClockCalibration c;
    c.interval_s = seconds;
    c.gpu_ticks = g1 - g0;
    c.uptime_ns = u1 - u0;
    c.ns_per_tick = c.gpu_ticks ? static_cast<double>(c.uptime_ns) / static_cast<double>(c.gpu_ticks) : 0;
    return c;
}

std::string thermal_state_name() {
    switch ([NSProcessInfo processInfo].thermalState) {
        case NSProcessInfoThermalStateNominal: return "nominal";
        case NSProcessInfoThermalStateFair: return "fair";
        case NSProcessInfoThermalStateSerious: return "serious";
        case NSProcessInfoThermalStateCritical: return "critical";
    }
    return "unknown";
}
