#!/usr/bin/env python3
"""
Python fastLowess benchmark runner with JSON output for comparison with statsmodels.

This is a standalone benchmark program that outputs results in JSON format
compatible with the Rust benchmark.
"""

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import fastLowess


# ============================================================================
# Constants
# ============================================================================

WARMUP_ITERATIONS = 3


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BenchmarkResult:
    name: str
    size: int
    iterations: int
    mean_time_ms: float = 0.0
    std_time_ms: float = 0.0
    median_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0

    def compute_stats(self, times: List[float]):
        if not times:
            return

        # Convert to milliseconds
        times_ms = [t * 1000.0 for t in times]

        self.mean_time_ms = np.mean(times_ms)
        self.std_time_ms = np.std(times_ms)
        self.median_time_ms = np.median(times_ms)
        self.min_time_ms = np.min(times_ms)
        self.max_time_ms = np.max(times_ms)


# ============================================================================
# Data Generation
# ============================================================================

def generate_data(size: int):
    x = np.array([i * 10.0 / size for i in range(size)])
    y = np.array([
        math.sin(xi) + math.sin(math.sin((i * 7.3) * 0.5)) * 0.2
        for i, xi in enumerate(x)
    ])
    return x, y


def generate_data_with_outliers(size: int):
    x = np.array([i * 10.0 / size for i in range(size)])
    y = np.array([math.sin(xi) for xi in x])

    # Add outliers (5% of points)
    n_outliers = max(size // 20, 1)
    for i in range(n_outliers):
        idx = (i * size) // n_outliers
        y[idx] += 3.0 if i % 2 == 0 else -3.0

    return x, y


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_basic_smoothing(sizes: List[int], iterations: int) -> List[BenchmarkResult]:
    results = []

    for size in sizes:
        print(f"Benchmarking basic smoothing with size={size}...")
        result = BenchmarkResult(f"basic_smoothing_{size}", size, iterations)

        x, y = generate_data(size)
        times = []

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            _ = fastLowess.smooth(x, y, fraction=0.3, iterations=3)

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            _ = fastLowess.smooth(x, y, fraction=0.3, iterations=3)
            times.append(time.perf_counter() - start)

        result.compute_stats(times)
        print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
        results.append(result)

    return results


def benchmark_fraction_variations(size: int, iterations: int) -> List[BenchmarkResult]:
    results = []
    fractions = [0.1, 0.2, 0.3, 0.5, 0.67, 0.8]

    x, y = generate_data(size)

    for frac in fractions:
        print(f"Benchmarking fraction={frac}...")
        result = BenchmarkResult(f"fraction_{frac}", size, iterations)
        times = []

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            _ = fastLowess.smooth(x, y, fraction=frac, iterations=3)

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            _ = fastLowess.smooth(x, y, fraction=frac, iterations=3)
            times.append(time.perf_counter() - start)

        result.compute_stats(times)
        print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
        results.append(result)

    return results


def benchmark_robustness_iterations(size: int, iterations: int) -> List[BenchmarkResult]:
    results = []
    niter_values = [0, 1, 2, 3, 5, 10]

    x, y = generate_data_with_outliers(size)

    for niter in niter_values:
        print(f"Benchmarking robustness iterations={niter}...")
        result = BenchmarkResult(f"iterations_{niter}", size, iterations)
        times = []

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            _ = fastLowess.smooth(x, y, fraction=0.3, iterations=niter)

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            _ = fastLowess.smooth(x, y, fraction=0.3, iterations=niter)
            times.append(time.perf_counter() - start)

        result.compute_stats(times)
        print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
        results.append(result)

    return results


def benchmark_delta_parameter(size: int, iterations: int) -> List[BenchmarkResult]:
    results = []

    x = np.array([i * 0.1 for i in range(size)])
    y = np.array([math.sin(xi) for xi in x])

    delta_configs = [
        ("delta_none", 0.0),
        ("delta_auto", None),  # Use None (auto)
        ("delta_small", 1.0),
        ("delta_large", 10.0),
    ]

    for name, delta_val in delta_configs:
        print(f"Benchmarking {name} (delta={delta_val})...")
        result = BenchmarkResult(name, size, iterations)
        times = []

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            if delta_val is None:
                _ = fastLowess.smooth(x, y, fraction=0.3, iterations=2)
            else:
                _ = fastLowess.smooth(x, y, fraction=0.3, iterations=2, delta=delta_val)

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            if delta_val is None:
                _ = fastLowess.smooth(x, y, fraction=0.3, iterations=2)
            else:
                _ = fastLowess.smooth(x, y, fraction=0.3, iterations=2, delta=delta_val)
            times.append(time.perf_counter() - start)

        result.compute_stats(times)
        print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
        results.append(result)

    return results


def benchmark_pathological_cases(size: int, iterations: int) -> List[BenchmarkResult]:
    results = []

    # Clustered x values
    print("Benchmarking clustered_x...")
    x_clustered = np.array([
        (i // 100) + (i % 100) * 1e-6
        for i in range(size)
    ])
    y_clustered = np.array([math.sin(xi) for xi in x_clustered])

    result = BenchmarkResult("clustered_x", size, iterations)
    times = []

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = fastLowess.smooth(x_clustered, y_clustered, fraction=0.5, iterations=2)

    for _ in range(iterations):
        start = time.perf_counter()
        _ = fastLowess.smooth(x_clustered, y_clustered, fraction=0.5, iterations=2)
        times.append(time.perf_counter() - start)

    result.compute_stats(times)
    print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    results.append(result)

    # Extreme outliers
    print("Benchmarking extreme_outliers...")
    x_normal = np.array([i * 10.0 / size for i in range(size)])
    y_outliers = np.array([math.sin(xi) for xi in x_normal])
    for i in range(0, size, 50):
        y_outliers[i] += 100.0 if i % 100 == 0 else -100.0

    result = BenchmarkResult("extreme_outliers", size, iterations)
    times = []

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = fastLowess.smooth(x_normal, y_outliers, fraction=0.3, iterations=5)

    for _ in range(iterations):
        start = time.perf_counter()
        _ = fastLowess.smooth(x_normal, y_outliers, fraction=0.3, iterations=5)
        times.append(time.perf_counter() - start)

    result.compute_stats(times)
    print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    results.append(result)

    # Constant y values
    print("Benchmarking constant_y...")
    y_constant = np.full(size, 5.0)

    result = BenchmarkResult("constant_y", size, iterations)
    times = []

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = fastLowess.smooth(x_normal, y_constant, fraction=0.3, iterations=2)

    for _ in range(iterations):
        start = time.perf_counter()
        _ = fastLowess.smooth(x_normal, y_constant, fraction=0.3, iterations=2)
        times.append(time.perf_counter() - start)

    result.compute_stats(times)
    print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    results.append(result)

    # High noise
    print("Benchmarking high_noise...")
    y_noisy = np.array([
        math.sin(xi / 10.0) * 0.1 + math.sin(math.sin((i * 7.3) * 0.5)) * 2.0
        for i, xi in enumerate(x_normal)
    ])

    result = BenchmarkResult("high_noise", size, iterations)
    times = []

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = fastLowess.smooth(x_normal, y_noisy, fraction=0.6, iterations=3)

    for _ in range(iterations):
        start = time.perf_counter()
        _ = fastLowess.smooth(x_normal, y_noisy, fraction=0.6, iterations=3)
        times.append(time.perf_counter() - start)

    result.compute_stats(times)
    print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    results.append(result)

    return results


def benchmark_realistic_scenarios(iterations: int) -> List[BenchmarkResult]:
    results = []
    size = 1000

    # Financial time series
    print("Benchmarking financial_timeseries...")
    x = np.array([float(i) for i in range(size)])
    y = np.array([
        xi * 0.01 + math.sin(xi / 50.0) * 0.5 + math.sin(math.sin((i * 7.3) * 0.5)) * 0.3
        for i, xi in enumerate(x)
    ])

    result = BenchmarkResult("financial_timeseries", size, iterations)
    times = []

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = fastLowess.smooth(x, y, fraction=0.1, iterations=2)

    for _ in range(iterations):
        start = time.perf_counter()
        _ = fastLowess.smooth(x, y, fraction=0.1, iterations=2)
        times.append(time.perf_counter() - start)

    result.compute_stats(times)
    print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    results.append(result)

    # Scientific data
    print("Benchmarking scientific_data...")
    x_sci = np.array([i * 0.01 for i in range(size)])
    y_sci = np.array([
        math.exp(xi * 2.0 * math.pi) * math.cos(xi * 10.0) + math.sin(math.sin((i * 13.7) * 0.3)) * 0.1
        for i, xi in enumerate(x_sci)
    ])

    result = BenchmarkResult("scientific_data", size, iterations)
    times = []

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = fastLowess.smooth(x_sci, y_sci, fraction=0.2, iterations=3)

    for _ in range(iterations):
        start = time.perf_counter()
        _ = fastLowess.smooth(x_sci, y_sci, fraction=0.2, iterations=3)
        times.append(time.perf_counter() - start)

    result.compute_stats(times)
    print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    results.append(result)

    # Genomic methylation
    print("Benchmarking genomic_methylation...")
    x_genomic = np.array([float(i * 1000) for i in range(size)])
    y_genomic = np.array([
        max(0.0, min(1.0, 0.5 + math.sin(xi / 5000.0) * 0.2 + math.sin(math.sin((i * 17.3) * 0.3)) * 0.15))
        for i, xi in enumerate(x_genomic)
    ])

    result = BenchmarkResult("genomic_methylation", size, iterations)
    times = []

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = fastLowess.smooth(x_genomic, y_genomic, fraction=0.2, iterations=3, delta=100.0)

    for _ in range(iterations):
        start = time.perf_counter()
        _ = fastLowess.smooth(x_genomic, y_genomic, fraction=0.2, iterations=3, delta=100.0)
        times.append(time.perf_counter() - start)

    result.compute_stats(times)
    print(f"  Mean: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    results.append(result)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("PYTHON FASTLOWESS BENCHMARK SUITE")
    print("=" * 80 + "\n")

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    # Core benchmarks
    print("\n" + "=" * 80)
    print("CORE BENCHMARKS")
    print("=" * 80 + "\n")

    all_results["basic_smoothing"] = [
        asdict(r) for r in benchmark_basic_smoothing([100, 500, 1000, 5000, 10000], 10)
    ]

    all_results["fraction_variations"] = [
        asdict(r) for r in benchmark_fraction_variations(1000, 10)
    ]

    all_results["robustness_iterations"] = [
        asdict(r) for r in benchmark_robustness_iterations(1000, 10)
    ]

    all_results["delta_parameter"] = [
        asdict(r) for r in benchmark_delta_parameter(5000, 10)
    ]

    # Stress tests
    print("\n" + "=" * 80)
    print("STRESS TESTS")
    print("=" * 80 + "\n")

    all_results["pathological_cases"] = [
        asdict(r) for r in benchmark_pathological_cases(1000, 10)
    ]

    # Application scenarios
    print("\n" + "=" * 80)
    print("APPLICATION SCENARIOS")
    print("=" * 80 + "\n")

    all_results["realistic_scenarios"] = [
        asdict(r) for r in benchmark_realistic_scenarios(10)
    ]

    # Save results
    json_str = json.dumps(all_results, indent=2)

    # Determine output directory (parent's output dir: ../output)
    script_dir = Path(__file__).parent
    out_dir = script_dir.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "fastLowess_benchmark.json"
    with open(out_path, "w") as f:
        f.write(json_str)

    print("\n" + "=" * 80)
    print(f"Results saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
