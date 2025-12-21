# fastlowess

[![PyPI](https://img.shields.io/pypi/v/fastlowess.svg?style=flat-square)](https://pypi.org/project/fastlowess/)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastlowess.svg?style=flat-square)](https://pypi.org/project/fastlowess/)
[![Documentation Status](https://readthedocs.org/projects/fastlowess-py/badge/?version=latest)](https://fastlowess-py.readthedocs.io/en/latest/?badge=latest)

**High-performance parallel LOWESS (Locally Weighted Scatterplot Smoothing) for Python** — A high-level wrapper around the [`fastLowess`](https://github.com/thisisamirv/fastLowess) Rust crate that offers **12-3800x faster performance** than standard implementations while providing robust statistics, uncertainty quantification, and memory-efficient streaming.

## Features

- **Parallel by Default**: Multi-core regression fits via Rust's Rayon, achieving multiple orders of magnitude speedups on large datasets.
- **Robust Statistics**: MAD-based scale estimation and IRLS with Bisquare, Huber, or Talwar weighting.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Production-Ready**: Comprehensive error handling, numerical stability, and high-performance numerical core.

## Robustness Advantages

This implementation is **more robust than statsmodels** due to two key design choices:

### MAD-Based Scale Estimation

For robustness weight calculations, this crate uses **Median Absolute Deviation (MAD)** for scale estimation:

```text
s = median(|r_i - median(r)|)
```

In contrast, statsmodels uses median of absolute residuals:

```text
s = median(|r_i|)
```

**Why MAD is more robust:**

- MAD is a **breakdown-point-optimal** estimator—it remains valid even when up to 50% of data are outliers.
- The median-centering step removes asymmetric bias from residual distributions.
- MAD provides consistent outlier detection regardless of whether residuals are centered around zero.

### Boundary Padding

This crate applies **boundary policies** (Extend, Reflect, Zero) at dataset edges:

- **Extend**: Repeats edge values to maintain local neighborhood size.
- **Reflect**: Mirrors data symmetrically around boundaries.
- **Zero**: Pads with zeros (useful for signal processing).

statsmodels does not apply boundary padding, which can lead to:

- Biased estimates near boundaries due to asymmetric local neighborhoods.
- Increased variance at the edges of the smoothed curve.

### Gaussian Consistency Factor

For interval estimation (confidence/prediction), residual scale is computed using:

```text
sigma = 1.4826 * MAD
```

The factor 1.4826 = 1/Phi^-1(3/4) ensures consistency with the standard deviation under Gaussian assumptions.

## Performance Advantages

Benchmarked against Python's `statsmodels`. Achieves **12-3800x faster performance** across different tested scenarios. The parallel implementation ensures that even at extreme scales (100k points), processing remains sub-20ms.

### Summary

| Category         | Matched | Median Speedup | Mean Speedup |
| :--------------- | :------ | :------------- | :----------- |
| **Scalability**  | 5       | **577.4x**     | 1375.0x      |
| **Financial**    | 4       | **242.1x**     | 263.5x       |
| **Iterations**   | 6       | **438.1x**     | 426.0x       |
| **Pathological** | 4       | **381.6x**     | 373.4x       |
| **Scientific**   | 4       | **165.1x**     | 207.5x       |
| **Fraction**     | 6       | **336.8x**     | 364.9x       |
| **Genomic**      | 4       | **23.1x**      | 22.7x        |
| **Delta**        | 4       | **3.6x**       | 6.0x         |

### Top 10 Performance Wins

| Benchmark          | statsmodels | fastlowess | Speedup     |
| :----------------- | :---------- | :--------- | :---------- |
| scale_100000       | 43727.2ms   | 11.5ms     | **3808.9x** |
| scale_50000        | 11159.9ms   | 5.9ms      | **1901.4x** |
| scale_10000        | 663.1ms     | 1.1ms      | **577.4x**  |
| fraction_0.05      | 197.2ms     | 0.4ms      | **556.5x**  |
| financial_10000    | 497.1ms     | 1.0ms      | **518.8x**  |
| iterations_0       | 74.2ms      | 0.2ms      | **492.9x**  |
| clustered          | 267.8ms     | 0.6ms      | **472.9x**  |
| iterations_1       | 148.5ms     | 0.3ms      | **471.5x**  |
| scale_5000         | 229.9ms     | 0.5ms      | **469.0x**  |
| scientific_10000   | 777.2ms     | 1.7ms      | **464.7x**  |

## Installation

```bash
pip install fastlowess
```

## Quick Start

```python
import numpy as np
import fastlowess

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

# Basic smoothing (parallel by default)
result = fastlowess.smooth(x, y, fraction=0.3)

print(f"Smoothed values: {result.y}")
```

## Smoothing Parameters

```python
import fastlowess

fastlowess.smooth(
    x, y,
    # Smoothing span (0, 1]
    fraction=0.5,

    # Robustness iterations for outlier resistance
    iterations=3,

    # Interpolation threshold for performance optimization
    # None (default) auto-calculates based on data range
    delta=0.01,

    # Kernel function selection
    # Options: "tricube", "gaussian", "epanechnikov", "uniform", etc.
    weight_function="tricube",

    # Robustness method selection
    # Options: "bisquare", "huber", "talwar"
    robustness_method="bisquare",

    # Zero-weight fallback behavior
    # Options: "use_local_mean", "return_original", "return_none"
    zero_weight_fallback="use_local_mean",

    # Boundary handling (for edge effects)
    # Options: "extend", "reflect", "zero"
    boundary_policy="extend",

    # Uncertainty Quantification
    confidence_intervals=0.95,
    prediction_intervals=0.95,

    # Output selection
    return_diagnostics=True,
    return_residuals=True,
    return_robustness_weights=True,

    # Cross-validation (for automatic parameter selection)
    cv_fractions=[0.3, 0.5, 0.7],
    cv_method="kfold",
    cv_k=5,

    # Multi-threading (via Rust/Rayon)
    parallel=True
)
```

## Result Structure

The `smooth()` function returns a `LowessResult` object with the following properties:

```python
result.x                    # Sorted independent variable values (numpy array)
result.y                    # Smoothed dependent variable values (numpy array)
result.standard_errors      # Point-wise standard errors (if computed)
result.confidence_lower     # Lower bound of confidence interval
result.confidence_upper     # Upper bound of confidence interval
result.prediction_lower     # Lower bound of prediction interval
result.prediction_upper     # Upper bound of prediction interval
result.residuals            # Model residuals (y - y_fit)
result.robustness_weights   # Final weights used for outlier handling
result.diagnostics          # Detailed fit diagnostics (RMSE, R^2, AIC, etc.)
result.iterations_used      # Number of robustness iterations performed
result.fraction_used        # Smoothing fraction used (best if via CV)
result.cv_scores            # RMSE scores for each CV candidate
```

The `diagnostics` object contains: `rmse`, `mae`, `r_squared`, `aic`, `aicc`, `effective_df`, `residual_sd`.

## Execution Modes

### Streaming Processing

For datasets too large to fit in memory (processes in chunks):

```python
result = fastlowess.smooth_streaming(
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
result = fastlowess.smooth_online(
    x, y,
    fraction=0.2,
    window_capacity=100,
    update_mode="incremental" # or "full"
)
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)
- **Use CV** when uncertain

### Robustness Iterations

- **0**: Clean data, speed critical
- **1-2**: Light contamination
- **3**: Default, good balance (recommended)
- **4-5**: Heavy outliers
- **>5**: Diminishing returns

### Kernel Function

- **Tricube** (default): Best all-around, smooth, efficient
- **Epanechnikov**: Theoretically optimal MSE
- **Gaussian**: Very smooth, no compact support
- **Uniform**: Fastest, least smooth (moving average)

### Delta Optimization

- **None**: Small datasets (n < 1000)
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

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

Check [Validation](https://github.com/thisisamirv/fastlowess-py/tree/bench/validation) for more information. Small variations in results are expected due to differences in scale estimation and padding.

## Related Work

- [fastLowess (Rust core)](https://github.com/thisisamirv/fastLowess)
- [fastLowess-R (R wrapper)](https://github.com/thisisamirv/fastlowess-R)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Dual-licensed under **AGPL-3.0** (Open Source) or **Commercial License**.
Contact `<thisisamirv@gmail.com>` for commercial inquiries.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.
