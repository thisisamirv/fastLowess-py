Advanced Usage
==============

This section details the advanced features of ``fastlowess``, including robustness, adaptive optimization, and cross-validation.

Robust Smoothing
----------------

The package provides multiple methods for handling outliers. By default, it uses **Bisquare** weighting with 3 robustness iterations.

Robustness Methods
^^^^^^^^^^^^^^^^^^

`fastlowess` supports three methods for downweighting outliers:

+------------+-------------------------+---------------------------+
| Method     | Behavior                | Use Case                  |
+============+=========================+===========================+
| Bisquare   | Smooth downweighting    | General-purpose (default) |
+------------+-------------------------+---------------------------+
| Huber      | Linear beyond threshold | Moderate outliers         |
+------------+-------------------------+---------------------------+
| Talwar     | Hard threshold (0 or 1) | Extreme contamination     |
+------------+-------------------------+---------------------------+

.. code-block:: python

   from fastlowess import smooth

   result = smooth(
       x, y,
       iterations=5,
       robustness_method="talwar"
   )

Boundary Policy
---------------

Control how the smoother behaves at the edges of the dataset to avoid boundary bias.

*   **"extend"** (default): Extends the boundary values (recommended for preserving trends).
*   **"reflect"**: Reflects values around the boundary.
*   **"zero"**: Pads with zeros.

.. image:: _static/images/robust_vs_standard_lowess.svg
   :alt: Robust vs Standard LOWESS
   :align: center
   :width: 100%

.. code-block:: python

   # Stop iterations if max change < 1e-4
   result = smooth(x, y, iterations=10, auto_converge=1e-4)

Cross-Validation
----------------

Automatically select the best ``fraction`` from a list of candidates.

.. code-block:: python

   fractions = [0.1, 0.2, 0.3, 0.5, 0.7]
   result = smooth(
       x, y,
       cv_fractions=fractions,
       cv_method="loocv"  # or "kfold"
   )

   print(f"Optimal fraction: {result.fraction_used}")

Diagnostics
-----------

Enable ``return_diagnostics`` to get objective measures of fit quality.

.. code-block:: python

   result = smooth(x, y, return_diagnostics=True)
   diag = result.diagnostics

   print(f"R-Squared: {diag.r_squared:.4f}")
   print(f"RMSE: {diag.rmse:.4f}")

Diagnostic Reference
^^^^^^^^^^^^^^^^^^^^

+-----------------+----------------------------------------+
| Metric          | Description                            |
+=================+========================================+
| **RMSE**        | Root Mean Squared Error                |
+-----------------+----------------------------------------+
| **MAE**         | Mean Absolute Error                    |
+-----------------+----------------------------------------+
| **R-squared**   | Coefficient of Determination           |
+-----------------+----------------------------------------+
| **aic/aicc**    | Information Criteria (if applicable)   |
+-----------------+----------------------------------------+
| **eff_df**      | Effective Degrees of Freedom           |
+-----------------+----------------------------------------+

Zero-Weight Handling
--------------------

In sparse regions where a neighborhood contains no points (or weights sum to zero), you can control the fallback behavior:

*   **"use_local_mean"** (default): Uses the mean of the raw neighborhood.
*   **"return_original"**: Returns the original Y value for that point.
*   **"return_none"**: Returns NaN.

.. image:: _static/images/confidence_vs_prediction_intervals.svg
   :alt: Confidence vs Prediction Intervals
   :align: center
   :width: 100%

.. code-block:: python

    import fastlowess
    import numpy as np

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y = np.array([2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7])

    # Compute both Confidence and Prediction Intervals
    result = fastlowess.smooth(
        x, y,
        fraction=0.5,
        confidence_intervals=0.95,      # 95% CI for the mean
        prediction_intervals=0.95,      # 95% PI for observations
    )

    print(f"{'X':>6} {'Smoothed':>10} {'CI Lower':>10} {'CI Upper':>10}")
    print("-" * 40)
    for i in range(len(x)):
        # Access interval bounds (None check handled internally usually, but good practice)
        lower = result.confidence_lower[i]
        upper = result.confidence_upper[i]
        print(f"{x[i]:6.1f} {result.y[i]:10.2f} {lower:10.2f} {upper:10.2f}")

    # Output:
    #      X   Smoothed   CI Lower   CI Upper
    # ----------------------------------------
    #    1.0       2.02       1.26       2.78
    #    2.0       4.00       3.33       4.68
    #    3.0       6.00       5.17       6.83
    #    4.0       8.10       7.14       9.06
    #    5.0      10.04       8.96      11.12
    #    6.0      12.03      10.97      13.09
    #    7.0      13.90      13.17      14.63
    #    8.0      15.78      14.98      16.58

Kernel Selection
----------------

You can choose different kernel functions via the `weight_function` parameter to control how neighboring points are weighted.

.. code-block:: python

    fastlowess.smooth(x, y, weight_function="gaussian")

+--------------+------------+-------------------+----------------------------+
| Kernel       | Efficiency | Smoothness        | Use Case                   |
+==============+============+===================+============================+
| Tricube      | 0.998      | Very smooth       | **Default**, Best overall  |
+--------------+------------+-------------------+----------------------------+
| Epanechnikov | 1.000      | Smooth            | Theoretically optimal MSE  |
+--------------+------------+-------------------+----------------------------+
| Gaussian     | 0.961      | Infinitely smooth | Very smooth data           |
+--------------+------------+-------------------+----------------------------+
| Biweight     | 0.995      | Very smooth       | Alternative to Tricube     |
+--------------+------------+-------------------+----------------------------+
| Uniform      | 0.943      | None              | Fastest, moving average    |
+--------------+------------+-------------------+----------------------------+
| Triangle     | 0.989      | Moderate          | Simple, fast               |
+--------------+------------+-------------------+----------------------------+
| Cosine       | 0.999      | Smooth            | Alternative compact kernel |
+--------------+------------+-------------------+----------------------------+

*Efficiency = Asymptotic Mean Integrated Squared Error (AMISE) relative to Epanechnikov (1.0 = optimal)*

Cross-Validation (Automatic Parameter Selection)
------------------------------------------------

Choosing the right `fraction` (bandwidth) can be difficult. `fastlowess` can automatically select the optimal fraction using Cross-Validation.

.. code-block:: python

    import fastlowess
    import numpy as np

    # 1. Generate noisy data
    nx = np.arange(100, dtype=float)
    ny = 2 * nx + 1 + np.random.randn(100) * 5

    # 2. Run Cross-Validation
    # Test different fractions (bandwidths) to find the best fit
    result = fastlowess.smooth(
        nx, ny,
        cv_fractions=[0.2, 0.3, 0.5, 0.7],  # Candidates
        cv_method="kfold",                  # "kfold" (default) or "loocv"
        cv_k=5                              # 5-fold CV
    )
    optimal_fraction = result.fraction_used
    
    print(f"Optimal fraction selected: {optimal_fraction}")
    
    # Output:
    # Optimal fraction selected: 0.7
    # result.y is now smoothed using the optimal fraction


Zero Weight Handling
--------------------

In rare cases (e.g., extremely sparse data or inappropriate bandwidth), a point may have no neighbors with positive weights. You can control the behavior with `zero_weight_fallback`:

*   **"use_local_mean"** (default): Use the mean of the neighborhood.
*   **"return_original"**: Return the original y-value (no smoothing).
*   **"return_none"**: Return NaN (useful for filtering).

Boundary Policy (Edge Handling)
-------------------------------

LOWESS traditionally uses asymmetric windows at boundaries, which can introduce bias. The ``boundary_policy`` parameter pads the data before smoothing to enable centered windows:

*   **"extend"** (default): Pad with constant values (first/last y-value).
*   **"reflect"**: Mirror the data at boundaries.
*   **"zero"**: Pad with zeros.

.. code-block:: python

    # Use reflective padding for better edge handling
    result = fastlowess.smooth(x, y, boundary_policy="reflect")

Auto-Convergence
----------------

Automatically stop robustness iterations when the smoothed values converge.

.. code-block:: python

    # Stop when the maximum change between iterations is < 1e-6
    result = fastlowess.smooth(
        x, y, 
        iterations=20, 
        auto_converge=1e-6
    )
    print(f"Converged after {result.iterations_used} iterations")
