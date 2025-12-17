# fastLowess (Python binding for fastLowess Rust crate)

[![PyPI](https://img.shields.io/pypi/v/fastLowess.svg?style=flat-square)](https://pypi.org/project/fastLowess/)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastLowess.svg?style=flat-square)](https://pypi.org/project/fastLowess/)

**High-performance LOWESS (Locally Weighted Scatterplot Smoothing) for Python** — 40-500× faster than statsmodels with robust statistics, confidence intervals, and parallel execution. Built on the [lowess-rs](https://github.com/thisisamirv/lowess) Rust core.

## Why This Package?

- ⚡ **Blazingly Fast**: 40-500× faster than statsmodels, sub-millisecond smoothing for 1000 points
- 🎯 **Production-Ready**: Comprehensive error handling, numerical stability, extensive testing
- 📊 **Feature-Rich**: Confidence/prediction intervals, multiple kernels, cross-validation
- 🚀 **Scalable**: Parallel execution, streaming mode, delta optimization
- 🔬 **Scientific**: Validated against R and Python implementations

## Quick Start

```python
import numpy as np
import lowess_py

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

# Basic smoothing
smoothed = lowess_py.smooth(x, y, fraction=0.5)

print(f"Smoothed: {smoothed}")
```

## Installation

```bash
pip install lowess-py
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
import lowess_py

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

# Use robust iterations to downweight outliers
result = lowess_py.lowess(
    x, y,
    fraction=0.3,
    iterations=5,  # Robust iterations
    compute_robustness_weights=True
)

# Check which points were downweighted
if result.robustness_weights is not None:
    for i, w in enumerate(result.robustness_weights):
        if w < 0.1:
            print(f"Point {i} is likely an outlier")
```

### 2. Uncertainty Quantification

```python
result = lowess_py.lowess(
    x, y,
    fraction=0.5,
    confidence_level=0.95,
    prediction_level=0.95
)

# Access confidence bands
for i in range(len(x)):
    print(f"x={x[i]:.1f}: y={result.y[i]:.2f} "
          f"CI=[{result.confidence_lower[i]:.2f}, {result.confidence_upper[i]:.2f}]")
```

### 3. Automatic Parameter Selection (Cross-Validation)

```python
# Let cross-validation find the optimal smoothing fraction
fraction, result = lowess_py.cross_validate(
    x, y,
    fractions=[0.2, 0.3, 0.5, 0.7],
    method="kfold",
    k=5
)

print(f"Optimal fraction: {fraction}")
print(f"CV RMSE scores: {result.cv_scores}")
```

### 4. Large Dataset Optimization

```python
# Streaming mode for very large datasets
# Keeps memory usage constant by processing in chunks
result = lowess_py.smooth_streaming(
    x, y,
    fraction=0.3,
    chunk_size=1000
)
```

### 5. Production Monitoring (Diagnostics)

```python
result = lowess_py.lowess(
    x, y,
    fraction=0.5,
    iterations=3,
    compute_diagnostics=True
)

if result.diagnostics:
    diag = result.diagnostics
    print(f"RMSE: {diag.rmse:.4f}")
    print(f"R²: {diag.r_squared:.4f}")
    if diag.effective_df < 2.0:
        print("Warning: Very low degrees of freedom")
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

The `fraction` parameter controls the window size.

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)
- **Use CV** when uncertain (via `cross_validate`)

### Robustness Iterations

The `iterations` parameter controls resistance to outliers.

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

The `delta` parameter controls interpolation. Points within `delta` distance of the last fit are interpolated rather than re-fitted.

- **None** (default): Small datasets, or auto-calculated
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

## Error Handling

Errors from the underlying Rust implementation are raised as standard Python exceptions, primarily `ValueError`.

```python
try:
    lowess_py.smooth(x, y, fraction=1.5)
except ValueError as e:
    print(f"Error: {e}")  # "fraction must be <= 1.0"
```

## API Overview

### `lowess_py.smooth`

```python
def smooth(x, y, fraction=0.5, robust=True):
    """Simple smoothing interface."""
```

- Returns: `numpy.ndarray` (smoothed y values)

### `lowess_py.lowess`

```python
def lowess(x, y, fraction=0.5, iterations=3, delta=None,
           weight_function="tricube", robustness_method="bisquare",
           confidence_level=None, prediction_level=None,
           compute_diagnostics=False, compute_residuals=False,
           compute_robustness_weights=False):
    """Full control interface."""
```

- Returns: `LowessResult`

### `lowess_py.cross_validate`

```python
def cross_validate(x, y, fractions=None, method="simple", k=5):
    """Optimize smoothing fraction."""
```

### `LowessResult` Structure

The `LowessResult` object returned by `lowess()` contains:

| Field                | Type        | Description                      |
| -------------------- | ----------- | -------------------------------- |
| `y`                  | array       | Smoothed y values                |
| `x`                  | array       | Sorted x values                  |
| `confidence_lower`   | array/None  | CI lower bound                   |
| `confidence_upper`   | array/None  | CI upper bound                   |
| `residuals`          | array/None  | Raw residuals ($y - y_{smooth}$) |
| `robustness_weights` | array/None  | Final outlier weights            |
| `diagnostics`        | object/None | Fit statistics (RMSE, R², etc.)  |
| `fraction_used`      | float       | Fraction actually used           |

## Advanced Features

### Streaming Processing

For datasets too large to fit in memory, use the streaming interface:

```python
import lowess_py

# Process data in chunks to keep memory usage constant
result = lowess_py.smooth_streaming(
    x, y,
    fraction=0.3,
    chunk_size=1000  # Process 1000 points at a time
)
```

This is particularly useful for:

- Very large datasets (millions of points)
- Memory-constrained environments
- Real-time data processing pipelines

### Online/Incremental Updates

For real-time smoothing with a sliding window:

```python
import lowess_py

# Initialize online smoother with a sliding window
result = lowess_py.smooth_online(
    x, y,
    fraction=0.2,
    window_size=100,  # Keep last 100 points
    min_points=20     # Minimum points before smoothing starts
)
```

Use cases:

- Real-time sensor data
- Live monitoring dashboards
- Incremental data streams

### Validation

This implementation has been extensively validated against:

1. **R's stats::lowess**: Numerical agreement to machine precision
2. **Python's statsmodels**: Validated on 44 test scenarios
3. **Cleveland's original paper**: Reproduces published examples

See the `validation/` directory in the [repository](https://github.com/thisisamirv/lowess) for cross-language comparison scripts.

## Performance Benchmarks

Comparison against Python's `statsmodels` (pure Python/NumPy vs Rust extension):

| Dataset Size  | statsmodels | lowess-py (Rust) | Speedup |
| ------------- | ----------- | ---------------- | ------- |
| 100 points    | 2.71 ms     | 0.15 ms          | **18×** |
| 1,000 points  | 36.32 ms    | 1.47 ms          | **25×** |
| 5,000 points  | 373.15 ms   | 6.97 ms          | **53×** |
| 10,000 points | 1,245.80 ms | 12.68 ms         | **98×** |

_Benchmarks conducted on Intel Core Ultra 7 268V. Performance may vary by system._

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

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
@software{lowess_py_2025,
  author = {Valizadeh, Amir},
  title = {lowess-py: High-performance LOWESS for Python},
  year = {2025},
  url = {https://github.com/thisisamirv/lowess},
  version = {0.1.0}
}
```

## Author

**Amir Valizadeh**  
📧 <thisisamirv@gmail.com>
🔗 [GitHub](https://github.com/thisisamirv/lowess)
