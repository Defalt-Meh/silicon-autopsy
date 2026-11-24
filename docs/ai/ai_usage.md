# AI usage log

This file records every contribution made by an AI assistant to this project,
as required by the course's academic-integrity rules.

- **Assistant:** Claude (Anthropic), model Claude Opus 5.5, used through Claude
  Code in the project repository.
- **Permission:** the student confirmed on 2026-11-24 that AI assistance,
  including implementation, is permitted for this project.
- **How to read the entries:** each entry lists what the AI produced and what
  the student decided, reviewed or verified. Entries are appended in date
  order and never rewritten.

---

## 2026-11-24 — Session 1: planning, measurement engine, validation

**Planning (before implementation was permitted)**
- Inspected the machine. Findings: Apple M4 Pro (16-core GPU, 24 GiB, 16 KiB
  pages), Metal counter sampling only at stage boundaries, Python 3.15 without
  scientific packages.
- Proposed:
  - the experimental design;
  - the statistical formulation: differential latency estimator, MLE, MoM and
    Normal-Inverse-Gamma Bayesian estimators for lognormal and gamma
    distributions, bootstrap-calibrated KS test, Welch test on session means,
    OLS with cluster-robust standard errors, segmented regression and
    change-point detection with session-block bootstrap;
  - the architecture;
  - the validation strategy and the requirement checklist.

**Implementation (after permission)**

The AI wrote:
- the C++/Objective-C++ engine (`src/`): CLI, Metal context, workloads and
  CPU references, config parser and schedule, CSV/JSON recording;
- the Metal kernels (`kernels/`);
- the CMake build;
- the engine unit tests (`tests/test_engine.cpp`);
- `configs/smoke.cfg` and `configs/discovery.cfg`;
- `docs/measurement.md`, `requirements.txt` and `.gitignore`.

**Measurement problems found and fixed by the AI during validation**

These were found from runs on this machine, which were not part of the
dataset.
1. GPU cache contents did not persist across command buffers. The trial was
   redesigned to use a single command buffer.
2. Chains shorter than one traversal measured a cold-cache quantity. Such
   cells are now skipped instead of measured.
3. GPU clock drops occurred after long latency-bound trials. A per-trial
   spin-up and three ALU clock references were added.
4. Mid-range working sets (256 KiB–4 MiB) showed per-encoder state
   variability. The AI tested and rejected two explanations: preceding
   workload history, and GPU interleaving (inter-encoder gaps are now
   recorded). It then changed the primary dependent variable from the
   differential estimate to the single long encoder's time per access, kept
   the differential as a bias check, and lengthened timed launches to up to
   8 traversals.

The evidence is summarized in `docs/measurement.md` §3.

**Not yet done by the student at the time of this entry**
- Review of the code.
- Re-running validation under the final collection conditions.

**External references used:** none beyond Apple's Metal API documentation for
API semantics.
