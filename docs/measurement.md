# Measurement definitions and validation

This document defines exactly how every recorded quantity is produced. It also
records the validation evidence that motivated the measurement design. All
numbers quoted in §3 are **preliminary engineering observations**. They were
collected on battery power with Low Power Mode enabled, and they are not part
of the experimental dataset. The validation protocol must be re-run under the
final collection conditions, and those results replace the numbers here.

## 1. Units of observation

| Level | What it is | Independent? |
|---|---|---|
| **Session** | One process run of `silicon-autopsy benchmark` with a recorded seed, at a distinct time. It has a fresh buffer allocation (new physical pages) and a fresh randomized schedule. | Yes: sessions are the independent replication unit. |
| **Trial** | One row of the CSV: one cell (configuration) measured once inside a session. It uses one command buffer and its own random permutation, from the per-trial seed `trial_seed`. | Conditionally: trials share a session (allocation, thermal history, background load). Analyses must treat the session as a block or cluster. |
| **Access** | One dependent load inside a kernel (10⁴–10⁷ per trial). | **No.** Never counted as observations; they are averaged inside the trial. |

The sample size for any inferential procedure is the number of trials, or of
sessions when the estimand is session-level. It is never the number of
accesses.

## 2. Trial structure and recorded columns

Each trial is one Metal command buffer containing seven compute encoders. Each
encoder holds one dispatch:

```
[0] clock spin-up      ALU chain, spin_n steps
[1] ALU reference      ALU chain, alu_ref_n steps        -> alu_ns_per_op_start
[2] warm-up            chase n_warm loads | bandwidth 1 pass -> gpu_ns_warmup
[3] timed A            short or long (random order, long_first)
[4] ALU reference                                         -> alu_ns_per_op_mid
[5] timed B            long or short
[6] ALU reference                                         -> alu_ns_per_op_end
```

GPU time for each encoder is the difference between two GPU timestamp counter
samples at the start and end of that encoder. They are taken with
`MTLCounterSamplingPointAtStageBoundary`, the only sampling point this GPU
supports, which means individual dispatches inside an encoder cannot be
timed. Ticks are converted to nanoseconds with `ns_per_tick`, which is
calibrated at the start and end of every session against `CLOCK_UPTIME_RAW`
over 0.2 s (stored in the session JSON). It has been measured at 0.99999 on
this machine.

### Pointer chase (`kind = chase`)

- **Chain:** the working set is divided into `n_slots` = ⌊W_req / stride⌋
  slots, `stride_bytes` apart. `working_set_bytes` = n_slots × stride is the
  address span actually used. The first 32-bit word of each slot holds the
  index of the next slot.
  - `random`: a uniformly random single cycle through all slots (Sattolo's
    algorithm, seeded by `trial_seed`).
  - `sequential`: slot k → k+1 mod n_slots.
- **Kernel:** one GPU thread executes `i = buf[i]` n times, then writes the
  final `i` to memory. That write prevents dead-code elimination, and the
  value is checked bit-exactly against the CPU.
- **Lengths:**

  n_short = max(n_min, ⌈min_traversals·S⌉, min(⌈target_traversals·S⌉, n_budget))

  where S = `n_slots`. Then `iteration_count_short` = n_short,
  `iteration_count` (long) = long_ratio × n_short, and `warmup_iterations` =
  max(n_min, S), which is one full traversal.
  - Every timed launch covers at least one full traversal (hard rule), and up
    to 8 traversals where the budget allows.
  - Cells that would need `iteration_count > n_max` are skipped and reported,
    never measured with a shorter chain.
- **`gpu_ns_short`, `gpu_ns_long`:** GPU time of the two timed encoders.
- **`effective_latency_ns`** (primary dependent variable) = gpu_ns_long / iteration_count.
  - **What it is:** the time per dependent access of a single encoder that
    started warm, because the same command buffer already traversed the
    whole chain.
  - **What it includes:** the few loop and addressing instructions on the
    dependent path. It is therefore an *effective* latency, not a pure
    load-to-use latency.
  - **What it doesn't include:** the per-encoder overhead (1.7–5.6 µs, §3.2),
    which contributes < 0.2% at n ≥ 128 K.
  - **Why it's primary:** it is one physical measurement. It cannot combine
    two encoders that ran in different cache states (§3.5).
- **`short_latency_ns`** = gpu_ns_short / iteration_count_short. This is a
  second per-encoder value from the **same trial**, so it is not an
  independent observation.
- **`differential_latency_ns`** = (gpu_ns_long − gpu_ns_short) / (iteration_count − iteration_count_short).
  - Under T(n) = a + b·n, this cancels any start-up cost a that both encoders
    share.
  - Comparing it with `effective_latency_ns` measures the residual start-up
    bias of the primary variable per cell.
  - It is unreliable whenever the two encoders ran in different states (§3.5).
- **`gpu_execution_ns`** = gpu_ns_long.

### Bandwidth (`kind = bandwidth`)

- **Kernel:** 65 536 threads (`thread_count`) in threadgroups of 256
  (`threadgroup_size`) read the buffer as 16-byte vectors with a grid-stride
  loop, so consecutive threads read consecutive vectors. Each thread does
  `iteration_count` passes over the buffer.
- **Accumulator:** a rotate-then-add accumulator makes the result depend on
  load order, so the compiler cannot fold the passes. The result is reduced
  with a SIMD sum plus an atomic add, and checked bit-exactly against the CPU.
- **Bytes:** `bytes_short` = passes_short × W and `bytes_long` = passes_long × W.
- **`bandwidth_gbps`** (primary) = bytes_long / gpu_ns_long, in bytes per ns,
  which equals decimal GB/s (10⁹ B/s).
- **`short_bandwidth_gbps`** = bytes_short / gpu_ns_short.
- **`differential_bandwidth_gbps`** = (bytes_long − bytes_short) / (gpu_ns_long − gpu_ns_short).
  This is a bias check, like the differential latency.

### Clock and overhead covariates

- **`alu_ns_per_op_{start,mid,end}`:** the duration of an `alu_ref_n`-step
  dependent integer chain, divided by `alu_ref_n`. It depends only on the GPU
  core clock, so it serves as a **frequency proxy** obtained without any
  vendor data.
- **`cmdbuf_ns_trial`:** `GPUEndTime − GPUStartTime` of the whole command
  buffer, on the GPU timeline.
- **`cpu_wall_ns_trial`:** CPU time from just before `commit` to the return of
  `waitUntilCompleted`.
- **`gpu_gap_ns_before_a`, `gpu_gap_ns_before_b`, `gpu_gap_ns_max`:** idle
  time on the GPU timeline between the end of one encoder and the start of
  the next, from the timestamp counters. A large gap would mean other GPU
  work ran between our encoders. Observed values are 0.3–0.7 µs.
- **Overhead split:**
  - `cmdbuf_ns_trial` − Σ(encoder GPU times) = GPU-side inter-encoder overhead.
  - `cpu_wall_ns_trial` − `cmdbuf_ns_trial` = CPU dispatch plus completion
    overhead.
- **`thermal_state`:** `NSProcessInfo.thermalState` (nominal, fair, serious or
  critical) after the trial.

### Validation and integrity

- **`validation_ok`** = 1 only when every encoder's result word matches the CPU
  reference bit-exactly. That covers the spin-up, all ALU references, the
  warm-up and both timed encoders.
- **Failed trials are written and flagged, never dropped.** `validation_note`
  says what failed.
- **Raw files are append-only and never overwritten.** The CSV is opened with
  `fopen(..., "wx")`.

### Session metadata (`<session_id>.json`)

The JSON file records:
- the seed and the verbatim configuration text;
- start and end UTC times;
- the clock calibration at session start and end;
- device properties;
- OS version and build, and hardware model;
- memory and page size;
- power source and Low Power Mode;
- the git commit and whether the working tree was clean;
- host and Metal compiler versions;
- trial counts, including invalid trials.

## 3. Validation evidence for the design (preliminary)

Validation is run with `silicon-autopsy validate`, which writes to
`data/validation/`. It records individual timed encoders rather than
differential trials.

### 3.1 Cache contents do not persist across command buffers

`persist_same_cb` vs `persist_separate_cb`: one traversal (n = n_slots) is
timed immediately after a warm-up traversal. The two are either in the same
command buffer, or in two command buffers committed one after the other.

| W | same command buffer (ns/access) | separate command buffer (ns/access) |
|---|---|---|
| 256 KiB | 96–121 | 229–245 |
| 16 MiB | 293–317 | 295–316 |
| 128 MiB | 341.5 | 341.5 |

A scratch probe gave the same result for two command buffers committed
back-to-back with no CPU wait in between (≈ 228 vs 96 ns at 64 KiB–2 MiB). So
the effect is tied to the **command-buffer boundary, not to GPU idle time**.
The mechanism (cache invalidation or flush, or some other scheduling effect)
is not identified. **Consequence:** warm-up and timed encoders must share one
command buffer. A design with separate launches measures a cold-cache quantity
at intermediate working sets.

### 3.2 Linearity T(n) = a + b·n

`linearity` mode: n ∈ {0, 2¹⁰ … 2²²}, 3 permutation seeds × 3 repetitions,
with W ∈ {4 KiB, 256 KiB, 16 MiB, 128 MiB}.

- **Overhead:** encoder overhead at n = 0 is 1.7–5.6 µs.
- **Transient:** at 256 KiB, T(n)/n falls from 97 ns (n = 4096 = one traversal)
  to an asymptote of 69.9 ns. This is a fixed transient a ≈ 110 µs, which is
  exactly what the differential estimator cancels.
- **Reproducibility:** at 256 KiB and 128 MiB, results agree across seeds and
  repetitions to within ±1 ns.
- **Exception:** at 16 MiB, per-access time varies by ±5% across seeds and
  repetitions. This W lies in a transition region (§3.4).

### 3.3 GPU clock (DVFS) is not stationary

The GPU clock was often low at the start of a trial that followed a long,
latency-bound trial (≥ 16 MiB working sets, 0.3–0.7 s of a single
memory-stalled thread). In those trials the ALU reference read 13.4–32 ns per
step instead of 12.7. Those trials were exactly the outliers, including one
negative differential latency. Adding the ≈ 25 ms spin-up encoder removed the
problem. With the same seed and the same schedule:
- sentinel spread: 42.6–47.3 ns before → **42.76–42.84 ns** after;
- cells at 256 KiB–2 MiB: up to 4× disagreement between repetitions before →
  < 0.6% after.

The three ALU references per trial remain as covariates. A clock-stability
criterion can then be applied as a **pre-specified sensitivity analysis**,
never as silent exclusion.

### 3.4 History does not explain it; the GPU does not interleave other work

- **Preconditioning has no effect:** running a 64 MiB chase or a bandwidth
  burst before a 1 MiB trial, either in the same command buffer or in the
  previous one, does not change that trial's latency.
- **No interleaving:** inter-encoder gaps are 0.3–0.7 µs in every trial, so no
  other GPU work ran between our encoders.

### 3.5 Per-encoder state variability and the choice of estimator

**Observation.** Eight identical consecutive encoders on the same 1 MiB chain,
inside one command buffer, read either ≈ 95.9 ns (the steady state) or
≈ 120–130 ns per access. In one run this alternated. In a later run almost
all encoders read 95.9. The slow value fits one extra traversal at ≈ 212 ns,
i.e. an encoder that started without the warm data available to it. The
mechanism (for example, where a single-thread dispatch is scheduled) cannot
be observed through Metal and is not claimed.

**Consequence.** When the short and long encoders of a trial land in
different states, the differential estimator is off by up to
≈ 116 ns × S / ΔN. Before the redesign this produced values from −51 to
+290 ns at 1–4 MiB.

**Design decisions.**
1. The primary variable is the single-encoder `effective_latency_ns`.
2. Timed launches cover up to 8 traversals, which dilutes any start-up
   transient. At 1 MiB the per-encoder latency is 96 ns, against 95.7 ns for
   the differential.
3. The differential is kept as a per-cell bias check.

**Residual variability.** At 2–4 MiB the per-encoder values spread
*continuously*: 96–164 ns at 2 MiB and 116–282 ns at 4 MiB. Elsewhere they
are tight (±1%). This region lies between two plateaus, where a cache level
is only partly effective. The spread is treated as a property of the system,
to be characterized by the distribution analysis, not removed.

**Full-scale check.** The same seed and schedule were run under the first and
final designs (945 trials each, 0 validation failures in both).
- Sentinel standard deviation: 0.444 → 0.085 ns.
- Median within-cell CV of E1 at 256 KiB–5.6 MiB: 24% → 2.1%.
- Non-positive latency estimates: 4 → 0.
- Bias check: the median |differential / effective − 1| across cells is
  0.23% (90th percentile 1.6%).

The remaining high-CV cells each contain a single slow trial among otherwise
identical ones (for example 95.9, 95.9 and 229 ns at 1.4 MiB). This is the
intermittent slow-encoder state, now confined to one-sided, right-tail
outliers.

## 4. Controlled vs uncontrolled factors

| Factor | Control |
|---|---|
| Compiler elimination | Every result is written to memory and validated bit-exactly. |
| Cold caches / start-up | Warm-up traversal in the same command buffer. Timed launches cover ≥ 1 and up to 8 traversals. The differential bias check is recorded per trial. |
| Dispatch overhead | < 0.2% of the primary variable. Measured separately (`cmdbuf`, `cpu_wall`, inter-encoder gaps, validation at n = 0). |
| GPU clock (DVFS) | Session warm-up, per-trial spin-up, three ALU clock references per trial. |
| Drift (thermal, background) | Fully randomized trial order. A fixed sentinel cell every 10 trials. Thermal state recorded. |
| Permutation-specific effects | A new random permutation per trial. |
| Physical page placement | Uncontrolled. Varies across sessions (fresh allocation), so it is absorbed by the session effect. |
| Display / WindowServer GPU work | **Uncontrolled.** The built-in display is driven by the same GPU. Minimize by closing apps and not using the machine during collection. |
| CPU / other agents on the shared memory system | **Uncontrolled.** Same mitigation. |
| GPU clock *within* an encoder | Only bracketed by the references, not measured continuously. |
| Which GPU core runs a single-thread dispatch | **Uncontrolled** and unobservable in Metal. |

## 5. Collection conditions (required for real data)

1. AC power, Low Power Mode **off** (`pmset -g`). Both are recorded and warned about.
2. Committed code (clean git tree). The commit hash is recorded and the tool warns otherwise.
3. Quit other applications, leave the machine untouched, and keep the display
   in a constant state.
4. Run each session as a separate invocation, at different times of day or on
   different days.
