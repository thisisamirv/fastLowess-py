fastlowess: High-performance LOWESS for Python
==============================================

**High-performance parallel LOWESS (Locally Weighted Scatterplot Smoothing) for Python**

A production-ready implementation built on top of a highly optimized Rust core. It offers **12-3800x speed improvements** over standard implementations while providing robust statistics, uncertainty quantification, and memory-efficient streaming.

What is LOWESS?
---------------

LOWESS (Locally Weighted Scatterplot Smoothing) is a nonparametric regression method that fits smooth curves through scatter plots. At each point, it fits a weighted polynomial (typically linear) using nearby data points.

.. image:: _static/images/lowess_smoothing_concept.svg
   :alt: LOWESS Smoothing Concept
   :align: center
   :width: 800

Key advantages:
*   **No parametric assumptions**: Adapts to the data's local structure.
*   **Robustness**: Handles outliers using iterative re-weighted least squares (IRLS).
*   **Uncertainty Quantification**: Provides confidence and prediction intervals.
*   **Scale**: Handles millions of points with parallel execution and streaming.

How LOWESS Works
----------------

LOWESS creates smooth curves through scattered data using local weighted neighborhoods:

1. For each point, select nearby neighbors (controlled by ``fraction``).
2. Fit a weighted polynomial (closer points get higher weight).
3. Use the fitted value as the smoothed estimate.
4. Optionally iterate to downweight outliers (robustness).

Robustness Advantages
---------------------

This implementation is **more robust than statsmodels** due to:

MAD-Based Scale Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^

We use **Median Absolute Deviation (MAD)** for scale estimation, which is breakdown-point-optimal:

.. math::

   s = \text{median}(|r_i - \text{median}(r)|)

Boundary Padding
^^^^^^^^^^^^^^^^

We apply **boundary policies** (Extend, Reflect, Zero) at dataset edges to maintain symmetric local neighborhoods, preventing the edge bias common in other implementations.

Gaussian Consistency Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For precision in intervals, residual scale is computed using:

.. math::

   \hat{\sigma} = 1.4826 \times \text{MAD}

Performance Advantages
----------------------

Benchmarked against Python's ``statsmodels``. Achieves **8.5x to 2800x faster performance** across different tested scenarios. The parallel implementation ensures that even at extreme scales (100k points), processing remains sub-20ms.

+------------------+---------+----------------+--------------+
| Category         | Matched | Median Speedup | Mean Speedup |
+==================+=========+================+==============+
| **Scalability**  | 5       | **283.2x**     | 922.0x       |
+------------------+---------+----------------+--------------+
| **Pathological** | 4       | **355.5x**     | 355.0x       |
+------------------+---------+----------------+--------------+
| **Iterations**   | 6       | **302.3x**     | 339.8x       |
+------------------+---------+----------------+--------------+
| **Fraction**     | 6       | **265.8x**     | 285.0x       |
+------------------+---------+----------------+--------------+
| **Financial**    | 4       | **176.7x**     | 215.2x       |
+------------------+---------+----------------+--------------+
| **Scientific**   | 4       | **201.1x**     | 225.6x       |
+------------------+---------+----------------+--------------+
| **Genomic**      | 4       | **17.5x**      | 18.6x        |
+------------------+---------+----------------+--------------+
| **Delta**        | 4       | **4.1x**       | 6.1x         |
+------------------+---------+----------------+--------------+

Top 10 Performance Wins
^^^^^^^^^^^^^^^^^^^^^^^

+-------------------+-------------+-------------+------------+
| Benchmark         | statsmodels | fastlowess  | Speedup    |
+===================+=============+=============+============+
| scale_100000      | 27.71s      | 9.9ms       | **2799.5x**|
+-------------------+-------------+-------------+------------+
| scale_50000       | 7.15s       | 5.7ms       | **1252.0x**|
+-------------------+-------------+-------------+------------+
| iterations_0      | 48.5ms      | 0.1ms       | **488.0x** |
+-------------------+-------------+-------------+------------+
| financial_10000   | 337.8ms     | 0.7ms       | **471.6x** |
+-------------------+-------------+-------------+------------+
| scientific_10000  | 522.4ms     | 1.2ms       | **432.5x** |
+-------------------+-------------+-------------+------------+
| clustered         | 172.2ms     | 0.4ms       | **426.1x** |
+-------------------+-------------+-------------+------------+
| constant_y        | 141.2ms     | 0.4ms       | **379.6x** |
+-------------------+-------------+-------------+------------+
| fraction_0.05     | 130.9ms     | 0.4ms       | **370.5x** |
+-------------------+-------------+-------------+------------+
| iterations_2      | 149.6ms     | 0.4ms       | **362.2x** |
+-------------------+-------------+-------------+------------+
| tricube           | 188.9ms     | 0.6ms       | **335.3x** |
+-------------------+-------------+-------------+------------+

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples
   advanced_usage
   execution_modes
   parameters

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
