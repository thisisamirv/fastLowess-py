# fastLowess

[![PyPI](https://img.shields.io/pypi/v/fastLowess.svg?style=flat-square)](https://pypi.org/project/fastLowess/)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastLowess.svg?style=flat-square)](https://pypi.org/project/fastLowess/)
[![Documentation Status](https://readthedocs.org/projects/fastlowess-py/badge/?version=latest)](https://fastlowess-py.readthedocs.io/en/latest/?badge=latest)

**High-performance parallel LOWESS (Locally Weighted Scatterplot Smoothing) for Python** — A high-level wrapper around the [`fastLowess`](https://github.com/thisisamirv/fastLowess) Rust crate that offers **4-3800x faster performance** than standard implementations while providing robust statistics, uncertainty quantification, and memory-efficient streaming.

## Features

- **Parallel by Default**: Multi-core regression fits via Rust's Rayon, achieving multiple orders of magnitude speedups on large datasets.
- **Robust Statistics**: MAD-based scale estimation and IRLS with Bisquare, Huber, or Talwar weighting.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Production-Ready**: Comprehensive error handling, numerical stability, and high-performance numerical core.

## Robustness Advantages

This implementation is **more robust than statsmodels** due to:

### MAD-Based Scale Estimation

We use **Median Absolute Deviation (MAD)** for scale estimation, which is breakdown-point-optimal:

$$s = \text{median}(|r_i - \text{median}(r)|)$$

### Boundary Padding

We apply **boundary policies** (Extend, Reflect, Zero) at dataset edges to maintain symmetric local neighborhoods, preventing the edge bias common in other implementations.

### Gaussian Consistency Factor

For precision in intervals, residual scale is computed using:

$$\hat{\sigma} = 1.4826 \times \text{MAD}$$

## Performance Advantages

Benchmarked against Python's `statsmodels`. Achieves **50-3800x faster performance** across different tested scenarios. The parallel implementation ensures that even at extreme scales (100k points), processing remains sub-20ms.

### Summary

| Category         | Matched | Median Speedup | Mean Speedup |
| :--------------- | :------ | :------------- | :----------- |
| **Scalability**  | 5       | **765x**       | 1433x        |
| **Pathological** | 4       | **448x**       | 416x         |
| **Iterations**   | 6       | **436x**       | 440x         |
| **Fraction**     | 6       | **424x**       | 413x         |
| **Financial**    | 4       | **336x**       | 385x         |
| **Scientific**   | 4       | **327x**       | 366x         |
| **Genomic**      | 4       | **20x**        | 25x          |
| **Delta**        | 4       | **4x**         | 5.5x         |

### Top 10 Performance Wins

| Benchmark          | statsmodels | fastLowess | Speedup   |
| :----------------- | :---------- | :--------- | :-------- |
| scale_100000       | 43.727s     | 11.4ms     | **3824x** |
| scale_50000        | 11.160s     | 5.95ms     | **1876x** |
| scale_10000        | 663.1ms     | 0.87ms     | **765x**  |
| financial_10000    | 497.1ms     | 0.66ms     | **748x**  |
| scientific_10000   | 777.2ms     | 1.07ms     | **729x**  |
| fraction_0.05      | 197.2ms     | 0.37ms     | **534x**  |
| scale_5000         | 229.9ms     | 0.44ms     | **523x**  |
| fraction_0.1       | 227.9ms     | 0.45ms     | **512x**  |
| financial_5000     | 170.9ms     | 0.34ms     | **497x**  |
| scientific_5000    | 268.5ms     | 0.55ms     | **489x**  |

## Installation

```bash
pip install fastLowess
```

## Quick Start

```python
import numpy as np
import fastLowess

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

# Basic smoothing (parallel by default)
result = fastLowess.smooth(x, y, fraction=0.5)

print(f"Smoothed values: {result.y}")
```

## Common Use Cases

### 1. Robust Smoothing (Handle Outliers)

```python
# Use robust iterations to downweight outliers
result = fastLowess.smooth(
    x, y,
    fraction=0.7,
    iterations=5,  # Robust iterations
    robustness_method="bisquare",  # "bisquare", "huber", or "talwar"
    return_robustness_weights=True
)

# Identify outliers (weights < 0.1)
outliers = np.where(result.robustness_weights < 0.1)[0]
```

### 2. Uncertainty Quantification

```python
result = fastLowess.smooth(
    x, y,
    fraction=0.5,
    confidence_intervals=0.95,
    prediction_intervals=0.95
)

# Access confidence bands
print(f"CI Lower: {result.confidence_lower}")
print(f"CI Upper: {result.confidence_upper}")
```

### 3. Automatic Parameter Selection (Cross-Validation)

```python
# Automatic selection of the best smoothing fraction
result = fastLowess.smooth(
    x, y,
    cv_fractions=[0.2, 0.3, 0.5, 0.7],  # Test these candidates
    cv_method="kfold",                  # "kfold" or "loocv"
    cv_k=5,
)

print(f"Optimal fraction used: {result.fraction_used}")
```

## Execution Modes

### Streaming Processing

For datasets too large to fit in memory (n > 1M):

```python
result = fastLowess.smooth_streaming(
    x, y,
    fraction=0.3,
    chunk_size=5000,
    overlap=500,
    parallel=True
)
```

### Online Processing

For real-time data streams or sliding windows:

```python
result = fastLowess.smooth_online(
    x, y,
    fraction=0.2,
    window_capacity=100,
    min_points=3,
    update_mode="incremental" # or "full"
)
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (Cleveland's choice)
- **Use CV** (via `cv_fractions`) when uncertain

### Robustness Iterations

- **0**: Clean data, speed critical
- **1-2**: Light contamination
- **3**: Default, good balance (recommended)
- **4-5**: Heavy outliers

### Delta Optimization

- **None**: Small datasets (n < 1000)
- **0.01 × range(x)**: Good starting point for dense data

## Documentation

For full documentation, API reference, and advanced features, visit [fastlowess-py.readthedocs.io](https://fastlowess-py.readthedocs.io/).

## Examples

Check the `examples` directory for advanced usage:

```bash
python examples/batch_smoothing.py
python examples/online_smoothing.py
python examples/streaming_smoothing.py
```

## Validation

Validated against:

- **Python (statsmodels)**: Passed on 44 distinct test scenarios.
- **Original Paper**: Reproduces Cleveland (1979) results.

Check [Validation](https://github.com/thisisamirv/fastLowess-py/tree/bench/validation) for more information. Small variations in results are expected due to differences in scale estimation and padding.

## Related Work

- [lowess (Rust core)](https://github.com/thisisamirv/lowess)
- [fastLowess (R wrapper)](https://github.com/thisisamirv/fastlowess-R)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Dual-licensed under **AGPL-3.0** (Open Source) or **Commercial License**.
Contact `<thisisamirv@gmail.com>` for commercial inquiries.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.
