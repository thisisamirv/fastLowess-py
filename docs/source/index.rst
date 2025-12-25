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

Benchmarked against Python's ``statsmodels``. Achieves **12-3800x faster performance** across different tested scenarios. The parallel implementation ensures that even at extreme scales (100k points), processing remains sub-20ms.

+------------------+---------+----------------+--------------+
| Category         | Matched | Median Speedup | Mean Speedup |
+==================+=========+================+==============+
| **Scalability**  | 5       | **577.4x**     | 1375.0x      |
+------------------+---------+----------------+--------------+
| **Pathological** | 4       | **381.6x**     | 373.4x       |
+------------------+---------+----------------+--------------+
| **Iterations**   | 6       | **438.1x**     | 426.0x       |
+------------------+---------+----------------+--------------+
| **Fraction**     | 6       | **336.8x**     | 364.9x       |
+------------------+---------+----------------+--------------+
| **Financial**    | 4       | **242.1x**     | 263.5x       |
+------------------+---------+----------------+--------------+
| **Scientific**   | 4       | **165.1x**     | 207.5x       |
+------------------+---------+----------------+--------------+
| **Genomic**      | 4       | **23.1x**      | 22.7x        |
+------------------+---------+----------------+--------------+
| **Delta**        | 4       | **3.6x**       | 6.0x         |
+------------------+---------+----------------+--------------+

Top 10 Performance Wins
^^^^^^^^^^^^^^^^^^^^^^^

+-------------------+-------------+-------------+------------+
| Benchmark         | statsmodels | fastlowess  | Speedup    |
+===================+=============+=============+============+
| scale_100000      | 43727.2ms   | 11.5ms      | **3808.9x**|
+-------------------+-------------+-------------+------------+
| scale_50000       | 11159.9ms   | 5.9ms       | **1901.4x**|
+-------------------+-------------+-------------+------------+
| scale_10000       | 663.1ms     | 1.1ms       | **577.4x** |
+-------------------+-------------+-------------+------------+
| fraction_0.05     | 197.2ms     | 0.4ms       | **556.5x** |
+-------------------+-------------+-------------+------------+
| financial_10000   | 497.1ms     | 1.0ms       | **518.8x** |
+-------------------+-------------+-------------+------------+
| iterations_0      | 74.2ms      | 0.2ms       | **492.9x** |
+-------------------+-------------+-------------+------------+
| clustered         | 267.8ms     | 0.6ms       | **472.9x** |
+-------------------+-------------+-------------+------------+
| iterations_1      | 148.5ms     | 0.3ms       | **471.5x** |
+-------------------+-------------+-------------+------------+
| scale_5000        | 229.9ms     | 0.5ms       | **469.0x** |
+-------------------+-------------+-------------+------------+
| scientific_10000  | 777.2ms     | 1.7ms       | **464.7x** |
+-------------------+-------------+-------------+------------+

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   advanced_usage
   execution_modes

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
