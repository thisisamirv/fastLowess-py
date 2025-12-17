#!/usr/bin/env python3
"""
fastLowess Batch Smoothing Examples

This example demonstrates batch LOWESS smoothing features:
- Basic smoothing with different parameters
- Robustness iterations for outlier handling
- Confidence and prediction intervals
- Diagnostics and cross-validation

The batch adapter (lowess function) is the primary interface for
processing complete datasets that fit in memory.
"""

import numpy as np
import time
import fastLowess


def main():
    print("=" * 80)
    print("fastLowess Batch Smoothing Examples")
    print("=" * 80)
    print()

    example_1_basic_smoothing()
    example_2_with_intervals()
    example_3_robust_smoothing()
    example_4_cross_validation()


def example_1_basic_smoothing():
    """Example 1: Basic Smoothing
    Demonstrates the fundamental smoothing workflow
    """
    print("Example 1: Basic Smoothing")
    print("-" * 80)

    # Generate synthetic dataset
    n = 10_000
    x = np.arange(n, dtype=float)
    y = np.sin(x * 0.1) + np.cos(x * 0.01)

    start = time.perf_counter()
    result = fastLowess.smooth(
        x, y,
        fraction=0.5,       # Use 50% of data for each local fit
        iterations=3,       # 3 robustness iterations
    )
    duration = time.perf_counter() - start

    print(f"Processed {n} points in {duration:.4f}s")
    print(f"First 5 smoothed values: {result.y[:5]}")
    print()


def example_2_with_intervals():
    """Example 2: Confidence and Prediction Intervals
    Demonstrates computing uncertainty intervals
    """
    print("Example 2: Confidence and Prediction Intervals")
    print("-" * 80)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y = np.array([2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7])

    result = fastLowess.smooth(
        x, y,
        fraction=0.5,
        confidence_intervals=0.95,      # 95% confidence intervals
        prediction_intervals=0.95,      # 95% prediction intervals
    )

    print(f"{'X':>8} {'Y_smooth':>12} {'CI_Lower':>12} {'CI_Upper':>12}")
    print("-" * 50)
    for i in range(len(x)):
        ci_lower = result.confidence_lower[i] if result.confidence_lower is not None else 0
        ci_upper = result.confidence_upper[i] if result.confidence_upper is not None else 0
        print(f"{x[i]:8.2f} {result.y[i]:12.4f} {ci_lower:12.4f} {ci_upper:12.4f}")
    print()


def example_3_robust_smoothing():
    """Example 3: Robust Smoothing with Outliers
    Demonstrates outlier handling with robustness iterations
    """
    print("Example 3: Robust Smoothing with Outliers")
    print("-" * 80)

    # Data with outliers
    n = 1000
    x = np.arange(n, dtype=float) * 0.1
    y = np.sin(x.copy())
    # Add periodic outliers
    y[::100] = x[::100] + 10.0

    methods = ["bisquare", "huber", "talwar"]

    for method in methods:
        result = fastLowess.smooth(
            x, y,
            fraction=0.1,
            iterations=3,
            robustness_method=method,
            return_robustness_weights=True,
        )
        
        if result.robustness_weights is not None:
            outliers = np.sum(np.array(result.robustness_weights) < 0.1)
            print(f"{method.capitalize()}: Identified {outliers} potential outliers (weight < 0.1)")
        else:
            print(f"{method.capitalize()}: Completed (weights not available)")
    print()


def example_4_cross_validation():
    """Example 4: Cross-Validation for Fraction Selection
    Demonstrates automatic parameter selection
    """
    print("Example 4: Cross-Validation for Fraction Selection")
    print("-" * 80)

    # Generate test data
    np.random.seed(42)
    x = np.arange(100, dtype=float)
    y = 2 * x + 1 + np.random.randn(100) * 5

    fractions = [0.2, 0.3, 0.5, 0.7]
    
    optimal_fraction, result = fastLowess.cross_validate(
        x, y,
        fractions=fractions,
        cv_method="kfold",
        cv_k=5,
    )

    print(f"Selected fraction: {optimal_fraction}")
    if result.cv_scores is not None:
        print(f"CV scores: {result.cv_scores}")
    print()


if __name__ == "__main__":
    main()
