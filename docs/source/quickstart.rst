Quick Start
===========

This guide will get you up and running with `fastLowess` in minutes.

Basic Usage
-----------

The primary function is `fastLowess.smooth()`. It takes x and y arrays and returns a `LowessResult` object containing the smoothed values.

.. code-block:: python

    import numpy as np
    import fastLowess
    import matplotlib.pyplot as plt

    # 1. Generate synthetic data with noise
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, 100)

    # 2. Perform smoothing
    # fraction=0.3 means 30% of data is used for each local regression
    result = fastLowess.smooth(x, y, fraction=0.3)

    # 3. Access the results
    print(f"Number of points: {len(result.y)}")
    print(f"First 5 smoothed values: {result.y[:5]}")

    # Output:
    # Number of points: 100
    # First 5 smoothed values: [0.3686 0.4011 0.4324 0.4628 0.4923]

    # (Optional) Plotting
    # plt.scatter(x, y, alpha=0.3, label="Noisy Data")
    # plt.plot(x, result.y, color='red', label="LOWESS Fit")
    # plt.legend()
    # plt.show()

Core Parameters
---------------

**Fraction (Smoothing Span)**

The `fraction` parameter controls the window size (bandwidth) as a proportion of the dataset.

.. image:: _static/images/fraction_effect_comparison.svg
   :alt: Fraction Effect Comparison
   :align: center
   :width: 100%

*   **Range**: `(0, 1]`
*   **0.1 - 0.3**: Captures fine details (wiggly curve). Best for data with high-frequency variation.
*   **0.4 - 0.6**: Balanced smoothing (default is 0.67). General purpose.
*   **> 0.7**: Captures global trends, smoothing out most local variation.

**Iterations (Robustness)**

Number of robust re-weighting iterations to handle outliers.

*   **0**: No robustness (fastest). Use for clean data.
*   **1-3**: Light to moderate robustness (recommended). Default is 3.
*   **4-6**: Strong robustness for heavily contaminated data.

Getting Diagnostics
-------------------

You can request fit statistics like R-squared ($R^2$), RMSE, and AIC:

.. code-block:: python

    result = fastLowess.smooth(x, y, fraction=0.3, return_diagnostics=True)

    if result.diagnostics:
        print(f"R-squared: {result.diagnostics.r_squared:.4f}")
        print(f"RMSE: {result.diagnostics.rmse:.4f}")

    # Output:
    # R-squared: 0.9012
    # RMSE: 0.2164

Next Steps
----------

*   Learn about handling outliers and uncertainty in :doc:`advanced_usage`.
*   Working with massive datasets? Check out :doc:`execution_modes`.
