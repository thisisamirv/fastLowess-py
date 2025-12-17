#!/usr/bin/env python3
"""
fastLowess Online Smoothing Examples

This example demonstrates online LOWESS smoothing for real-time data:
- Basic incremental processing with streaming data
- Real-time sensor data smoothing
- Different window sizes and their effects
- Memory-bounded processing

The online adapter (smooth_online function) is designed for:
- Real-time data streams
- Memory-constrained environments
- Sensor data processing
- Incremental updates without reprocessing entire dataset
"""

import numpy as np
import time
import math
import fastLowess


def main():
    print("=" * 80)
    print("fastLowess Online Smoothing Examples")
    print("=" * 80)
    print()

    example_1_basic_online()
    example_2_sensor_simulation()
    example_3_window_size_comparison()
    example_4_memory_bounded()


def example_1_basic_online():
    """Example 1: Basic Online Processing
    Demonstrates incremental data processing
    """
    print("Example 1: Basic Online Processing")
    print("-" * 80)

    # Simulate streaming data: y = 2x + 1 with small noise
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([3.1, 5.0, 7.2, 8.9, 11.1, 13.0, 15.2, 16.8, 19.1, 21.0])

    # Process all at once with online adapter
    result = fastLowess.smooth_online(
        x, y,
        fraction=0.5,
        window_capacity=5,
        min_points=3,
        iterations=2,
        weight_function="tricube",
        robustness_method="bisquare",
    )

    print("Processing data points with sliding window...")
    print(f"Window capacity: 5")
    print(f"Output points: {len(result.y)}")
    print(f"Smoothed values: {result.y}")
    print()


def example_2_sensor_simulation():
    """Example 2: Real-Time Sensor Data Simulation
    Simulates processing temperature sensor readings
    """
    print("Example 2: Real-Time Sensor Data Simulation")
    print("-" * 80)
    print("Simulating temperature sensor readings with noise...\n")

    # Simulate temperature sensor: base temp 20°C with daily cycle + noise
    n = 24  # 24 hours
    x = np.arange(n, dtype=float)
    base_temp = 20.0
    daily_cycle = 5.0 * np.sin(x * math.pi / 12.0)
    noise = np.array([(hour * 7) % 11 * 0.3 - 1.5 for hour in range(n)])
    y = base_temp + daily_cycle + noise

    result = fastLowess.smooth_online(
        x, y,
        fraction=0.4,
        window_capacity=12,  # Half-day window
        min_points=3,
        iterations=2,
    )

    print(f"{'Hour':>6} {'Raw Temp':>12} {'Smoothed':>12}")
    print("-" * 35)
    
    # Show first 10 values
    for i in range(min(10, len(result.y))):
        print(f"{x[i]:6.0f} {y[i]:12.2f}°C {result.y[i]:12.2f}°C")
    
    if len(result.y) > 10:
        print(f"  ... ({len(result.y) - 10} more rows)")
    print()


def example_3_window_size_comparison():
    """Example 3: Window Size Comparison
    Shows how different window sizes affect smoothing behavior
    """
    print("Example 3: Window Size Comparison")
    print("-" * 80)

    # Generate test data with some variation
    x = np.arange(1, 51, dtype=float)
    y = 2.0 * x + np.sin(x * 0.5) * 3.0

    window_sizes = [5, 10, 20]

    for window_size in window_sizes:
        result = fastLowess.smooth_online(
            x, y,
            fraction=0.5,
            window_capacity=window_size,
            min_points=3,
            iterations=2,
        )
        
        print(f"Window capacity: {window_size}")
        print(f"  Output points: {len(result.y)}")
        if len(result.y) >= 5:
            print(f"  Last 5 smoothed: {result.y[-5:]}")
        else:
            print(f"  Smoothed values: {result.y}")
    print()


def example_4_memory_bounded():
    """Example 4: Memory-Bounded Processing
    Demonstrates efficient processing for resource-constrained systems
    """
    print("Example 4: Memory-Bounded Processing")
    print("-" * 80)

    # Simulate a long data stream
    total_points = 10_000
    print(f"Processing {total_points} data points with minimal memory footprint...")

    x = np.arange(total_points, dtype=float)
    y = 2.0 * x + np.sin(x * 0.1) * 5.0 + (np.arange(total_points) % 7 - 3.0) * 0.5

    start = time.perf_counter()
    result = fastLowess.smooth_online(
        x, y,
        fraction=0.3,
        window_capacity=20,  # Small window = low memory usage
        min_points=3,
        iterations=1,
        parallel=False,  # Sequential for low latency
    )
    duration = time.perf_counter() - start

    print(f"\nProcessed {len(result.y)} points in {duration:.4f}s")
    if len(result.y) > 0:
        print(f"Final smoothed value: {result.y[-1]:.2f}")
    print(f"Memory usage: Constant (window size = 20 points)")
    print()


if __name__ == "__main__":
    main()
