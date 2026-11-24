# SILICON AUTOPSY

### Interrogating a GPU Through Statistical Inference

**Research question:** can observable properties of an undocumented GPU memory
hierarchy be reconstructed from execution-time measurements and statistical
inference alone?

SILICON AUTOPSY treats the GPU of an Apple Silicon Mac as a black box. It
measures dependent-load latency and streaming bandwidth with controlled Metal
microbenchmarks, then infers memory-performance regimes statistically. The
inference uses no manufacturer specifications.

> Status: the measurement engine is implemented and validated (preliminary).
> The statistical analysis (`analysis/`) is the next stage.

## Requirements

- macOS on Apple Silicon, with Xcode (Metal compiler) and CMake ≥ 3.20.
- Python 3.13 for the analysis:
  `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Build and test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build              # engine unit tests (no GPU needed)
./build/silicon-autopsy info        # device capabilities + GPU clock calibration
```

## Collecting data

First read the collection conditions in
[docs/measurement.md §5](docs/measurement.md): AC power, Low Power Mode off,
committed code, and an otherwise idle machine.

```bash
./build/silicon-autopsy validate                                   # validation protocol, about 4 min
./build/silicon-autopsy benchmark                                  # smoke test, about 15 s (configs/smoke.cfg)
./build/silicon-autopsy benchmark --config configs/discovery.cfg   # one discovery session
```

Each session writes `data/raw/<phase>/<session_id>.csv` (one row per trial)
and a `.json` metadata sidecar. The sidecar holds the seed, the config text,
the device, the OS, the power state, the git commit and the clock
calibration. Add `--dry-run` to see a configuration's size without running it.
Add `--seed N` to reproduce a schedule and its permutations exactly.

## Experimental phases

1. **Discovery:** several independent sessions of `configs/discovery.cfg`,
   used to identify candidate latency regimes.
2. **Freeze:** the candidate regimes, hypotheses and analysis plan are written
   down and committed *before* any confirmation data is collected.
3. **Confirmation:** independent sessions of a confirmation config, used only
   for the confirmatory tests.

## Repository layout

```text
src/        C++ / Objective-C++ benchmark engine
kernels/    Metal Shading Language kernels
configs/    experiment configurations (smoke, discovery, ...)
analysis/   Python statistical analysis
data/raw/   raw measurements (never edited); data/validation/ validation runs
results/    figures and tables
tests/      engine unit tests (statistical-method tests go here too)
docs/       measurement definitions, validation, AI-usage log
```

## Documentation

- [docs/measurement.md](docs/measurement.md): exact definition of every
  recorded quantity, the validation evidence behind the design, and
  limitations.
- [docs/ai/ai_usage.md](docs/ai/ai_usage.md): log of AI contributions.
