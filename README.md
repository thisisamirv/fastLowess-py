# fastlowess

[![PyPI](https://img.shields.io/pypi/v/fastlowess.svg?style=flat-square)](https://pypi.org/project/fastlowess/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastlowess.svg?style=flat-square)](https://pypi.org/project/fastlowess/)
[![Documentation Status](https://readthedocs.org/projects/fastlowess-py/badge/?version=latest)](https://fastlowess-py.readthedocs.io/en/latest/?badge=latest)
[![Conda](https://anaconda.org/conda-forge/fastlowess/badges/version.svg)](https://anaconda.org/conda-forge/fastlowess)

**High-performance parallel LOWESS (Locally Weighted Scatterplot Smoothing) for Python** — A high-level wrapper around the [`fastLowess`](https://github.com/thisisamirv/fastLowess) Rust crate that adds rayon-based parallelism and seamless NumPy integration.

## Features

- **Parallel by Default**: Multi-core regression fits via [rayon](https://crates.io/crates/rayon), achieving multiple orders of magnitude speedups on large datasets.
- **Robust Statistics**: MAD-based scale estimation and IRLS with Bisquare, Huber, or Talwar weighting.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Production-Ready**: Comprehensive error handling, numerical stability, and high-performance numerical core.

> [!IMPORTANT]
> **Full Documentation & API Reference:**
>
> ## 📖 [fastlowess-py.readthedocs.io](https://fastlowess-py.readthedocs.io/)

## Robustness Advantages

Built on the same core as `lowess`, this implementation is **more robust than statsmodels** due to two key design choices:

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
- **NoBoundary**: No padding (original Cleveland's LOWESS).

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

The `fastLowess` crate demonstrates massive performance gains over Python's `statsmodels`. The Rust CPU backend is the decisive winner across almost all standard benchmarks, often achieving **multi-hundred-fold speedups**.

The table below shows speedups relative to the **baseline**.

- **Standard Benchmarks**: Baseline is `statsmodels` (Python).
- **Large Scale Benchmarks**: Baseline is `Rust (Serial)` (1x), as `statsmodels` times out.

| Name                  | statsmodels |      R      |  Rust (CPU)*  | Rust (GPU)|
|-----------------------|-------------|-------------|---------------|-----------|
| clustered             |  162.77ms   |  [82.8x]²   |  [203-433x]¹  |   32.4x   |
| constant_y            |  133.63ms   |  [92.3x]²   |  [212-410x]¹  |   17.5x   |
| delta_large           |   0.51ms    |   [0.8x]²   |  [3.8-2.2x]¹  |   0.1x    |
| delta_medium          |   0.79ms    |   [1.3x]²   |  [4.4-3.4x]¹  |   0.1x    |
| delta_none            |  414.86ms   |    2.5x     |  [3.8-13x]²   | [63.5x]¹  |
| delta_small           |   1.45ms    |   [1.7x]²   |  [4.3-4.5x]¹  |   0.2x    |
| extreme_outliers      |  488.96ms   |  [106.4x]²  |  [201-388x]¹  |   28.9x   |
| financial_1000        |   13.55ms   |  [76.6x]²   |  [145-108x]¹  |   4.7x    |
| financial_10000       |  302.20ms   |  [168.3x]²  |  [453-611x]¹  |   26.3x   |
| financial_500         |   6.49ms    |  [58.0x]¹   |  [113-58x]²   |   2.7x    |
| financial_5000        |  103.94ms   |  [117.3x]²  |  [296-395x]¹  |   14.1x   |
| fraction_0.05         |  122.00ms   |  [177.6x]²  |  [421-350x]¹  |   14.5x   |
| fraction_0.1          |  140.59ms   |  [112.8x]²  |  [291-366x]¹  |   15.9x   |
| fraction_0.2          |  181.57ms   |  [85.3x]²   |  [210-419x]¹  |   19.3x   |
| fraction_0.3          |  220.98ms   |  [84.8x]²   |  [168-380x]¹  |   22.4x   |
| fraction_0.5          |  296.47ms   |  [80.9x]²   |  [146-415x]¹  |   27.3x   |
| fraction_0.67         |  362.59ms   |  [83.1x]²   |  [129-413x]¹  |   32.0x   |
| genomic_1000          |   17.82ms   |  [15.9x]²   |   [19-33x]¹   |   6.5x    |
| genomic_10000         |  399.90ms   |    3.6x     |  [5.3-16x]²   | [70.3x]¹  |
| genomic_5000          |  138.49ms   |    5.0x     |  [7.0-19x]²   | [34.8x]¹  |
| genomic_50000         |  6776.57ms  |    2.4x     |  [3.5-11x]²   | [269.2x]¹ |
| high_noise            |  435.85ms   |  [132.6x]²  |  [134-375x]¹  |   32.3x   |
| iterations_0          |   45.18ms   |  [128.4x]²  |  [266-405x]¹  |   10.6x   |
| iterations_1          |   94.10ms   |  [114.3x]²  |  [236-384x]¹  |   14.4x   |
| iterations_10         |  495.65ms   |  [116.0x]²  |  [204-369x]¹  |   27.0x   |
| iterations_2          |  135.48ms   |  [109.0x]²  |  [219-432x]¹  |   16.6x   |
| iterations_3          |  181.56ms   |  [108.8x]²  |  [213-382x]¹  |   18.7x   |
| iterations_5          |  270.58ms   |  [110.4x]²  |  [208-345x]¹  |   22.7x   |
| scale_1000            |   17.95ms   |  [82.6x]²   |  [150-107x]¹  |   8.1x    |
| scale_10000           |  408.13ms   |  [178.1x]²  |  [433-552x]¹  |   76.3x   |
| scale_5000            |  139.81ms   |  [133.6x]²  |  [289-401x]¹  |   28.8x   |
| scale_50000           |  6798.58ms  |  [661.0x]²  | [1077-1264x]¹ |  277.2x   |
| scientific_1000       |   19.04ms   |  [70.1x]²   |  [113-115x]¹  |   5.4x    |
| scientific_10000      |  479.57ms   |  [190.7x]²  |  [370-663x]¹  |   35.2x   |
| scientific_500        |   8.59ms    |  [49.6x]²   |   [91-52x]¹   |   3.2x    |
| scientific_5000       |  161.42ms   |  [124.9x]²  |  [244-427x]¹  |   17.9x   |
| scale_100000**        |      -      |      -      |    1-1.3x     |   0.3x    |
| scale_1000000**       |      -      |      -      |    1-1.3x     |   0.3x    |
| scale_2000000**       |      -      |      -      |    1-1.5x     |   0.3x    |
| scale_250000**        |      -      |      -      |    1-1.4x     |   0.3x    |
| scale_500000**        |      -      |      -      |    1-1.3x     |   0.3x    |

\* **Rust (CPU)**: Shows range `Seq - Par`. E.g., `12-48x` means 12x speedup (Sequential) and 48x speedup (Parallel). Rank determined by Parallel speedup.
\*\* **Large Scale**: `Rust (Serial)` is the baseline (1x).

¹ Winner (Fastest implementation)
² Runner-up (Second fastest implementation)

**Key Takeaways**:

1. **Rust (Parallel CPU)** is the dominant performer for general-purpose workloads, consistently achieving the highest speedups (often 300x-500x over statsmodels).
2. **R (stats::lowess)** is a very strong runner-up, frequently outperforming statsmodels by ~80-150x, but generally trailing Rust Parallel.
3. **Rust (GPU)** excels in specific high-compute scenarios (e.g., `genomic` with large datasets or `delta_none` where interpolation is skipped), but carries overhead that makes it slower than the highly optimized CPU backend for smaller datasets.
4. **Large Scale Scaling**: At very large scales (100k - 2M points), the parallel CPU backend maintains a modest lead (1.3x - 1.5x) over the sequential CPU backend, likely bottlenecked by memory bandwidth rather than compute.
5. **Small vs Large Delta**: Setting `delta=0` (no interpolation, `delta_none`) allows the GPU to shine (63.5x speedup), outperforming both CPU variants due to the massive O(N²) interaction workload being parallelized across thousands of GPU cores.

Check [Benchmarks for fastLowess](https://github.com/thisisamirv/fastLowess/tree/bench/benchmarks) for detailed results and reproducible benchmarking code.

## Installation

Install via PyPI:

```bash
pip install fastlowess
```

Or install from conda-forge:

```bash
conda install -c conda-forge fastlowess
```

## Quick Start

```python
import numpy as np
import fastlowess

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

# Basic smoothing (parallel CPU by default)
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

    # Robustness iterations
    iterations=3,

    # Interpolation threshold
    delta=0.01,

    # Kernel function
    weight_function="tricube",

    # Robustness method
    robustness_method="bisquare",

    # Zero-weight fallback
    zero_weight_fallback="use_local_mean",

    # Boundary handling
    boundary_policy="extend",

    # Intervals
    confidence_intervals=0.95,
    prediction_intervals=0.95,

    # Diagnostics
    return_diagnostics=True,
    return_residuals=True,
    return_robustness_weights=True,

    # Cross-validation
    cv_fractions=[0.3, 0.5, 0.7],
    cv_method="kfold",
    cv_k=5,

    # Convergence
    auto_converge=1e-4,

    # Parallelism
    parallel=True
)
```

## Result Structure

The `smooth()` function returns a `LowessResult` object:

```python
result.x                    # Sorted independent variable values
result.y                    # Smoothed dependent variable values
result.standard_errors      # Point-wise standard errors
result.confidence_lower     # Lower bound of confidence interval
result.confidence_upper     # Upper bound of confidence interval
result.prediction_lower     # Lower bound of prediction interval
result.prediction_upper     # Upper bound of prediction interval
result.residuals            # Residuals (y - fit)
result.robustness_weights   # Final robustness weights
result.diagnostics          # Diagnostics (RMSE, R^2, etc.)
result.iterations_used      # Number of iterations performed
result.fraction_used        # Smoothing fraction used
result.cv_scores            # CV scores for each candidate
```

## Streaming Processing

For datasets that don't fit in memory:

```python
result = fastlowess.smooth_streaming(
    x, y,
    fraction=0.3,
    chunk_size=5000,
    overlap=500,
    parallel=True
)
```

## Online Processing

For real-time data streams:

```python
result = fastlowess.smooth_online(
    x, y,
    fraction=0.2,
    window_capacity=100,
    update_mode="incremental" # or "full"
)
```

## Backend

> [!NOTE]
> A *beta* GPU backend is available for acceleration in the Rust crate, but it is not exposed in the Python API due to added dependencies and complexity. Feedbacks on if this is something you would like to see are welcome or how to expose it in a user-friendly way are appreciated.

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Local, captures rapid changes
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)

### Robustness Iterations

- **0**: Clean data, speed critical
- **1-3**: Default, good balance
- **4-5**: Heavy outliers

### Kernel Function

- **Tricube** (default): Best all-around
- **Epanechnikov**: Optimal MSE
- **Gaussian**: Very smooth
- **Uniform**: Moving average

### Delta Optimization

- **None**: Small datasets (n < 1000)
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

## Examples

Check the `examples` directory:

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

- [fastLowess (Rust core)](https://github.com/thisisamirv/fastLowess)
- [fastLowess-R (R wrapper)](https://github.com/thisisamirv/fastlowess-R)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of

- Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.
