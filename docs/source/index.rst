**High-performance LOWESS (Locally Weighted Scatterplot Smoothing) for Python**

`fastLowess` provides a lightning-fast, parallelized implementation of the LOWESS algorithm, built on top of a highly optimized Rust core. It offers 5-287× speed improvements over standard Python implementations while providing robust statistics, confidence intervals, and memory-efficient streaming for large datasets.

What is LOWESS?
---------------

LOWESS (Locally Weighted Scatterplot Smoothing) is a nonparametric regression method that fits smooth curves through scatter plots. At each point, it fits a weighted polynomial (typically linear) using nearby data points.

.. image:: _static/images/lowess_smoothing_concept.svg
   :alt: LOWESS Smoothing Concept
   :align: center
   :width: 800

Key advantages:
*   **No parametric assumptions**: Adapts to the data's structure.
*   **Robustness**: Can ignore outliers using iterative re-weighting.
*   **Uncertainty**: provides confidence and prediction intervals.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   advanced_usage
   execution_modes
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
