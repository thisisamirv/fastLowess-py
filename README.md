# fastLowess (Python binding for fastLowess Rust crate)

[![PyPI](https://img.shields.io/pypi/v/fastLowess.svg?style=flat-square)](https://pypi.org/project/fastLowess/)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastLowess.svg?style=flat-square)](https://pypi.org/project/fastLowess/)
[![Documentation Status](https://readthedocs.org/projects/fastlowess-py/badge/?version=latest)](https://fastlowess-py.readthedocs.io/en/latest/?badge=latest)

**High-performance LOWESS (Locally Weighted Scatterplot Smoothing) for Python** — 5-287× faster than statsmodels with robust statistics, confidence intervals, and parallel execution. Built on the [fastLowess](https://github.com/thisisamirv/fastLowess) Rust crate.

## Why This Package?

- ⚡ **Blazingly Fast**: 5-287× faster than statsmodels, sub-millisecond smoothing for 1000 points
- 🎯 **Production-Ready**: Comprehensive error handling, numerical stability, extensive testing
- 📊 **Feature-Rich**: Confidence/prediction intervals, multiple kernels, cross-validation
- 🚀 **Scalable**: Parallel execution, streaming mode, delta optimization
- 🔬 **Scientific**: Validated against R and Python implementations

## Quick Start

For full documentation including advanced usage, API reference, and examples, visit [fastlowess-py.readthedocs.io](https://fastlowess-py.readthedocs.io/).

```python
import numpy as np
import fastLowess

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

# Basic smoothing
result = fastLowess.smooth(x, y, fraction=0.5)

print(f"Smoothed: {result.y}")
print(f"Fraction used: {result.fraction_used}")
```

## Installation

```bash
pip install fastLowess
```

## Features at a Glance

| Feature                  | Description                             | Use Case                      |
| ------------------------ | --------------------------------------- | ----------------------------- |
| **Robust Smoothing**     | IRLS with Bisquare/Huber/Talwar weights | Outlier-contaminated data     |
| **Confidence Intervals** | Point-wise standard errors & bounds     | Uncertainty quantification    |
| **Cross-Validation**     | Auto-select optimal fraction            | Unknown smoothing parameter   |
| **Multiple Kernels**     | Tricube, Epanechnikov, Gaussian, etc.   | Different smoothness profiles |
| **Parallel Execution**   | Multi-threaded via Rust/Rayon           | Large datasets (n > 1000)     |
| **Streaming Mode**       | Constant memory usage                   | Very large datasets           |
| **Delta Optimization**   | Skip dense regions                      | 10× speedup on dense data     |

## Common Use Cases

### 1. Robust Smoothing (Handle Outliers)

```python
import numpy as np
import fastLowess

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.1, 100.0, 8.2, 9.8])  # Outlier at index 2

# Use robust iterations to downweight outliers
result = fastLowess.smooth(
    x, y,
    fraction=0.7,
    iterations=5,  # Robust iterations
    return_robustness_weights=True
)

# Check which points were downweighted
if result.robustness_weights is not None:
    for i, w in enumerate(result.robustness_weights):
        if w < 0.1:
            print(f"Point {i} is likely an outlier")
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
for i in range(len(x)):
    print(f"x={x[i]:.1f}: y={result.y[i]:.2f} "
          f"CI=[{result.confidence_lower[i]:.2f}, {result.confidence_upper[i]:.2f}]")
```

### 3. Automatic Parameter Selection (Cross-Validation)

```python
# Cross-validation is integrated into smooth() via cv_fractions
result = fastLowess.smooth(
    x, y,
    cv_fractions=[0.2, 0.3, 0.5, 0.7],  # Fractions to test
    cv_method="kfold",                   # "kfold" or "loocv"
    cv_k=5                               # Number of folds
)

print(f"Optimal fraction: {result.fraction_used}")
print(f"CV RMSE scores: {result.cv_scores}")
```

### 4. Large Dataset Optimization

```python
# Streaming mode for very large datasets
# Keeps memory usage constant by processing in chunks
result = fastLowess.smooth_streaming(
    x, y,
    fraction=0.3,
    chunk_size=5000,
    overlap=500
)
```

### 5. Production Monitoring (Diagnostics)

```python
result = fastLowess.smooth(
    x, y,
    fraction=0.5,
    iterations=3,
    return_diagnostics=True
)

if result.diagnostics:
    diag = result.diagnostics
    print(f"RMSE: {diag.rmse:.4f}")
    print(f"MAE: {diag.mae:.4f}")
    print(f"R²: {diag.r_squared:.4f}")
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

The `fraction` parameter controls the window size.

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)
- **Use CV** when uncertain (via `cv_fractions`)

### Robustness Iterations

The `iterations` parameter controls resistance to outliers.

- **0**: Clean data, speed critical
- **1-2**: Light contamination
- **3**: Default, good balance (recommended)
- **4-5**: Heavy outliers
- **>5**: Diminishing returns

### Kernel Function

The `weight_function` parameter controls the kernel.

- **"tricube"** (default): Best all-around, smooth, efficient
- **"epanechnikov"**: Theoretically optimal MSE
- **"gaussian"**: Very smooth, no compact support
- **"uniform"**: Fastest, least smooth (moving average)
- **"biweight"**: Similar to tricube
- **"triangle"**: Linear decay
- **"cosine"**: Smooth cosine weighting

### Delta Optimization

The `delta` parameter controls interpolation. Points within `delta` distance of the last fit are interpolated rather than re-fitted.

- **None** (default): Small datasets, or auto-calculated
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

## Error Handling

Errors from the underlying Rust implementation are raised as standard Python exceptions, primarily `ValueError`.

```python
try:
    fastLowess.smooth(x, y, fraction=1.5)
except ValueError as e:
    print(f"Error: {e}")  # "fraction must be <= 1.0"
```

## API Reference

### `fastLowess.smooth`

The primary interface for LOWESS smoothing. Processes the entire dataset in memory with optional parallel execution.

```python
def smooth(
    x, y,
    fraction=0.67,                    # Smoothing fraction (0, 1]
    iterations=3,                     # Robustness iterations
    delta=None,                       # Interpolation threshold
    weight_function="tricube",        # Kernel function
    robustness_method="bisquare",     # Outlier method
    confidence_intervals=None,        # CI level (e.g., 0.95)
    prediction_intervals=None,        # PI level (e.g., 0.95)
    return_diagnostics=False,         # Compute RMSE, R², etc.
    return_residuals=False,           # Include residuals
    return_robustness_weights=False,  # Include weights
    zero_weight_fallback="use_local_mean",
    auto_converge=None,               # Auto-convergence tolerance
    max_iterations=None,              # Max iterations (default: 20)
    cv_fractions=None,                # Fractions for CV
    cv_method="kfold",                # "kfold" or "loocv"
    cv_k=5                            # Folds for k-fold CV
) -> LowessResult
```

### `fastLowess.smooth_streaming`

Streaming LOWESS for large datasets. Processes data in chunks to maintain constant memory usage.

```python
def smooth_streaming(
    x, y,
    fraction=0.3,                     # Smoothing fraction
    chunk_size=5000,                  # Points per chunk
    overlap=None,                     # Overlap (default: 10%)
    iterations=3,                     # Robustness iterations
    weight_function="tricube",        # Kernel function
    robustness_method="bisquare",     # Outlier method
    parallel=True                     # Enable parallelism
) -> LowessResult
```

### `fastLowess.smooth_online`

Online LOWESS with sliding window for real-time data streams.

```python
def smooth_online(
    x, y,
    fraction=0.2,                     # Fraction within window
    window_capacity=100,              # Max points in window
    min_points=3,                     # Min points before smoothing
    iterations=3,                     # Robustness iterations
    weight_function="tricube",        # Kernel function
    robustness_method="bisquare",     # Outlier method
    parallel=False                    # Enable parallelism
) -> LowessResult
```

### `LowessResult` Structure

The `LowessResult` object returned by all functions contains:

| Field                | Type        | Description                         |
| -------------------- | ----------- | ----------------------------------- |
| `x`                  | array       | Sorted x values                     |
| `y`                  | array       | Smoothed y values                   |
| `fraction_used`      | float       | Fraction actually used              |
| `iterations_used`    | int/None    | Robustness iterations performed     |
| `standard_errors`    | array/None  | Standard errors (if CI/PI enabled)  |
| `confidence_lower`   | array/None  | CI lower bound                      |
| `confidence_upper`   | array/None  | CI upper bound                      |
| `prediction_lower`   | array/None  | PI lower bound                      |
| `prediction_upper`   | array/None  | PI upper bound                      |
| `residuals`          | array/None  | Raw residuals (y - y_smooth)        |
| `robustness_weights` | array/None  | Final outlier weights [0, 1]        |
| `diagnostics`        | object/None | Fit statistics (RMSE, R², etc.)     |
| `cv_scores`          | array/None  | CV scores for tested fractions      |

### `Diagnostics` Structure

| Field          | Type       | Description                      |
| -------------- | ---------- | -------------------------------- |
| `rmse`         | float      | Root Mean Squared Error          |
| `mae`          | float      | Mean Absolute Error              |
| `r_squared`    | float      | Coefficient of determination     |
| `residual_sd`  | float      | Residual standard deviation      |
| `aic`          | float/None | Akaike Information Criterion     |
| `aicc`         | float/None | Corrected AIC                    |
| `effective_df` | float/None | Effective degrees of freedom     |

## Advanced Features

### Streaming Processing

For datasets too large to fit in memory:

```python
import fastLowess

# Process data in chunks to keep memory usage constant
result = fastLowess.smooth_streaming(
    x, y,
    fraction=0.3,
    chunk_size=5000,
    overlap=500
)
```

Use cases:

- Very large datasets (millions of points)
- Memory-constrained environments
- Batch processing pipelines

### Online/Incremental Updates

For real-time smoothing with a sliding window:

```python
import fastLowess

# Initialize online smoother with a sliding window
result = fastLowess.smooth_online(
    x, y,
    fraction=0.2,
    window_capacity=100,  # Keep last 100 points
    min_points=3          # Minimum points before smoothing starts
)
```

Use cases:

- Real-time sensor data
- Live monitoring dashboards
- Incremental data streams

### Validation

This implementation has been extensively validated against:

1. **R's stats::lowess**: Numerical agreement to machine precision
2. **Python's statsmodels**: Validated on multiple test scenarios
3. **Cleveland's original paper**: Reproduces published examples

## Performance Benchmarks

Comparison against Python's `statsmodels` (pure Python/NumPy vs Rust extension):

| Dataset Size  | statsmodels | fastLowess | Speedup  |
| ------------- | ----------- | ---------- | -------- |
| 100 points    | 1.79 ms     | 0.13 ms    | **14×**  |
| 500 points    | 9.86 ms     | 0.26 ms    | **38×**  |
| 1,000 points  | 22.80 ms    | 0.39 ms    | **59×**  |
| 5,000 points  | 229.76 ms   | 2.04 ms    | **112×** |
| 10,000 points | 742.99 ms   | 2.59 ms    | **287×** |

_Benchmarks conducted on Intel Core Ultra 7 268V. Performance may vary by system._

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

See the [LICENSE](LICENSE) file for details.

## References

**Original papers:**

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". _Journal of the American Statistical Association_, 74(368): 829-836. [DOI:10.2307/2286407](https://doi.org/10.2307/2286407)
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots by Robust Locally Weighted Regression". _The American Statistician_, 35(1): 54.

**Related implementations:**

- [R stats::lowess](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lowess.html)
- [Python statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html)

## Citation

```bibtex
@software{fastLowess_2025,
  author = {Valizadeh, Amir},
  title = {fastLowess: High-performance LOWESS for Python},
  year = {2025},
  url = {https://github.com/thisisamirv/fastLowess-py},
  version = {0.1.0}
}
```

## Author

**Amir Valizadeh**  
📧 <thisisamirv@gmail.com>
🔗 [GitHub](https://github.com/thisisamirv/fastLowess-py)
