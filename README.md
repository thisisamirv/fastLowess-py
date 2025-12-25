# FastLowess Validation & Benchmarking Workspace

This workspace is dedicated to validating the correctness and benchmarking the performance of the [fastlowess](https://github.com/thisisamirv/fastLowess-py) python package against the reference Python implementation (`statsmodels`).

It installs the `fastlowess` package from the `develop` branch (git dependency) to ensure the latest changes are tested.

> [!IMPORTANT]
> Before running benchmarks or validation, run `make install` to install the latest version of the `fastlowess` package from the git develop branch.

## structure

- `benchmarks/`: Performance benchmarking suite.
- `validation/`: Correctness validation suite.

## How to Run Benchmarks

Benchmarks measure execution time across various scenarios (basic smoothing, robustness iterations, pathological cases, etc.).

### 1. Run FastLowess Benchmarks

```bash
python3 benchmarks/fastlowess/benchmark.py
```

*Output: `benchmarks/output/fastlowess_benchmark.json`*

### 2. Run Statsmodels Benchmarks

```bash
# form the root directory
python3 benchmarks/statsmodels/benchmark.py
```

*Output: `benchmarks/output/statsmodels_benchmark.json`*

### 3. Compare Benchmark Results

Generate a comparison report showing speedups and regressions.

```bash
cd benchmarks
python3 compare_benchmark.py
```

*See `benchmarks/INTERPRETATION.md` for analysis.*

## How to Run Validation

Validation ensures the `fastlowess` implementation produces results identical (or acceptable close) to `statsmodels`.

### 1. Run FastLowess Validation

```bash
python3 validation/fastlowess/validate.py
```

*Output: `validation/output/fastlowess_validate.json`*

### 2. Run Statsmodels Validation

```bash
# from the root directory
python3 validation/statsmodels/validate.py
```

*Output: `validation/output/statsmodels_validate.json`*

### 3. Compare Validation Results

Check for mismatches in smoothed values, residuals, and diagnostics.

```bash
cd validation
python3 compare_validation.py
```

*See `validation/INTERPRETATION.md` for analysis.*

## Requirements

- **Python**: 3.x with `numpy`, `scipy`, `statsmodels` installed.
