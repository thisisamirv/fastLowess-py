#!/usr/bin/env python3
"""
fastLowess Streaming Smoothing Examples

This example demonstrates streaming LOWESS smoothing for large datasets:
- Basic chunked processing
- Different chunk sizes and overlap strategies
- Processing very large datasets
- Parallel vs sequential execution

The streaming adapter (smooth_streaming function) is designed for:
- Large datasets (>100K points) that don't fit in memory
- Batch processing pipelines
- File-based data processing
- ETL (Extract, Transform, Load) workflows
"""

import numpy as np
import time
import fastLowess


def main():
    print("=" * 80)
    print("fastLowess Streaming Smoothing Examples")
    print("=" * 80)
    print()

    example_1_basic_streaming()
    example_2_chunk_size_comparison()
    example_3_large_dataset()
    example_4_parallel_comparison()


def example_1_basic_streaming():
    """Example 1: Basic Streaming Processing
    Demonstrates the fundamental streaming workflow
    """
    print("Example 1: Basic Streaming Processing")
    print("-" * 80)

    # Generate test data: y = 2x + 1 with noise
    n = 100
    x = np.arange(n, dtype=float)
    y = 2.0 * x + 1.0 + np.sin(x * 0.3) * 2.0

    # Process with streaming adapter
    result = fastLowess.smooth_streaming(
        x, y,
        fraction=0.5,
        chunk_size=30,
        overlap=10,
        iterations=2,
        weight_function="tricube",
        robustness_method="bisquare",
    )

    print(f"Dataset: {n} points")
    print(f"Chunk size: 30, Overlap: 10")
    print(f"Output points: {len(result.y)}")
    print(f"All points processed: {len(result.y) == n}")
    print(f"First 5 smoothed values: {result.y[:5]}")
    print()


def example_2_chunk_size_comparison():
    """Example 2: Chunk Size Comparison
    Shows how different chunk sizes affect processing
    """
    print("Example 2: Chunk Size Comparison")
    print("-" * 80)

    # Generate test data
    n = 500
    x = np.arange(n, dtype=float)
    y = 2.0 * x + 1.0

    chunk_configs = [
        (50, 10, "Small chunks"),
        (100, 20, "Medium chunks"),
        (200, 40, "Large chunks"),
    ]

    for chunk_size, overlap, description in chunk_configs:
        start = time.perf_counter()
        result = fastLowess.smooth_streaming(
            x, y,
            fraction=0.5,
            chunk_size=chunk_size,
            overlap=overlap,
            iterations=2,
        )
        duration = time.perf_counter() - start

        print(f"{description} (size: {chunk_size}, overlap: {overlap})")
        print(f"  Output points: {len(result.y)}")
        print(f"  Time: {duration:.4f}s")
    print()


def example_3_large_dataset():
    """Example 3: Large Dataset Processing
    Demonstrates processing a very large dataset
    """
    print("Example 3: Large Dataset Processing")
    print("-" * 80)

    n = 50_000  # 50K points
    print(f"Processing {n} data points in streaming mode...")

    x = np.arange(n, dtype=float)
    y = 2.0 * x + 1.0 + np.sin(x * 0.01) * 10.0

    start = time.perf_counter()
    result = fastLowess.smooth_streaming(
        x, y,
        fraction=0.3,
        chunk_size=5000,
        overlap=500,
        iterations=2,
        parallel=True,  # Enable parallel execution
    )
    duration = time.perf_counter() - start

    print(f"Processed {len(result.y)} points in {duration:.4f}s")
    print(f"Memory efficiency: Constant (chunk size = 5000 points)")
    print()


def example_4_parallel_comparison():
    """Example 4: Parallel vs Sequential Comparison
    Compares execution time with and without parallelism
    """
    print("Example 4: Parallel vs Sequential Comparison")
    print("-" * 80)

    n = 10_000
    x = np.arange(n, dtype=float)
    y = np.sin(x * 0.1) + np.cos(x * 0.01)

    # Parallel execution
    start = time.perf_counter()
    result_parallel = fastLowess.smooth_streaming(
        x, y,
        fraction=0.5,
        chunk_size=1000,
        overlap=100,
        iterations=3,
        parallel=True,
    )
    parallel_time = time.perf_counter() - start

    # Sequential execution
    start = time.perf_counter()
    result_sequential = fastLowess.smooth_streaming(
        x, y,
        fraction=0.5,
        chunk_size=1000,
        overlap=100,
        iterations=3,
        parallel=False,
    )
    sequential_time = time.perf_counter() - start

    print(f"Parallel:   {parallel_time:.4f}s ({len(result_parallel.y)} points)")
    print(f"Sequential: {sequential_time:.4f}s ({len(result_sequential.y)} points)")
    if sequential_time > 0:
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    print()


if __name__ == "__main__":
    main()
