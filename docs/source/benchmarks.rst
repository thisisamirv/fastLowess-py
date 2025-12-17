Benchmarks
==========

`fastLowess` is designed for performance. Below are benchmark results comparing it against `statsmodels.nonparametric.smoothers_lowess`.

Comparison
----------

Benchmarks were conducted on an Intel Core Ultra 7 268V.

+---------------+--------------+-------------------+-------------+
| Dataset Size  | statsmodels  | fastLowess (Rust) | Speedup     |
+===============+==============+===================+=============+
| 100 points    | 1.79 ms      | 0.13 ms           | **14x**     |
+---------------+--------------+-------------------+-------------+
| 500 points    | 9.86 ms      | 0.26 ms           | **38x**     |
+---------------+--------------+-------------------+-------------+
| 1,000 points  | 22.80 ms     | 0.39 ms           | **59x**     |
+---------------+--------------+-------------------+-------------+
| 5,000 points  | 229.76 ms    | 2.04 ms           | **112x**    |
+---------------+--------------+-------------------+-------------+
| 10,000 points | 742.99 ms    | 2.59 ms           | **287x**    |
+---------------+--------------+-------------------+-------------+

Performance Tips
----------------

1.  **Parallel Execution**: `fastLowess` uses Rayon for parallel execution. This provides significant speedups for N > 1,000. It is enabled by default for batch operations.
2.  **Delta Optimization**: For dense datasets, setting `delta` allows `fastLowess` to interpolate values, avoiding expensive re-computation.
3.  **Streaming**: For datasets that don't fit in CPU cache or RAM, `smooth_streaming` ensures constant memory usage.
