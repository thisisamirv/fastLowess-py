# Benchmark Interpretation (fastlowess)

## Summary

The `fastlowess` package demonstrates massive performance gains over Python's `statsmodels`, ranging from **8.5x to over 2700x** speedup. The addition of **parallel execution** (via Rust/Rayon) and optimized algorithm defaults makes it exceptionally well-suited for high-throughput data processing and large-scale datasets.

## Category Comparison

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Scalability**  | 5       | **283.2x**     | 922.0x       |
| **Pathological** | 4       | **355.5x**     | 355.0x       |
| **Iterations**   | 6       | **302.3x**     | 339.8x       |
| **Fraction**     | 6       | **265.8x**     | 285.0x       |
| **Financial**    | 4       | **176.7x**     | 215.2x       |
| **Scientific**   | 4       | **201.1x**     | 225.6x       |
| **Genomic**      | 4       | **17.5x**      | 18.6x        |
| **Delta**        | 4       | **4.1x**       | 6.1x         |

## Top 10 FastLowess Wins

| Benchmark        | statsmodels | fastlowess | Speedup     |
|------------------|-------------|------------|-------------|
| scale_100000     | 27713.2ms   | 9.9ms      | **2799.5x** |
| scale_50000      | 7146.4ms    | 5.7ms      | **1252.0x** |
| iterations_0     | 48.5ms      | 0.1ms      | **488.0x**  |
| financial_10000  | 337.8ms     | 0.7ms      | **471.6x**  |
| scientific_10000 | 522.4ms     | 1.2ms      | **432.5x**  |
| clustered        | 172.2ms     | 0.4ms      | **426.1x**  |
| constant_y       | 141.2ms     | 0.4ms      | **379.6x**  |
| fraction_0.05    | 130.9ms     | 0.4ms      | **370.5x**  |
| iterations_2     | 149.6ms     | 0.4ms      | **362.2x**  |
| tricube          | 188.9ms     | 0.6ms      | **335.3x**  |

## Regressions

**None identified.** `fastlowess` outperforms `statsmodels` in all matched benchmarks. The parallel implementation ensures that even at extreme scales (100k points), processing remains around 10ms.

## Detailed Results

### Scalability (1K - 100K points)

| Size    | fastlowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 2.29ms     | 19.6ms      | 8.6x    |
| 5,000   | 0.52ms     | 146.3ms     | 283x    |
| 10,000  | 1.57ms     | 419.9ms     | 267x    |
| 50,000  | 5.71ms     | 7146.4ms    | 1252x   |
| 100,000 | 9.90ms     | 27713.2ms   | 2800x   |

### Fraction Variations (n=5000)

| Fraction | fastlowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| 0.05     | 0.35ms     | 130.9ms     | 370x    |
| 0.10     | 0.61ms     | 154.1ms     | 251x    |
| 0.20     | 0.62ms     | 194.9ms     | 317x    |
| 0.30     | 0.99ms     | 238.4ms     | 240x    |
| 0.50     | 1.30ms     | 326.4ms     | 251x    |
| 0.67     | 1.44ms     | 403.7ms     | 281x    |

### Robustness Iterations (n=5000)

| Iterations | fastlowess | statsmodels | Speedup |
|------------|------------|-------------|---------|
| 0          | 0.10ms     | 48.5ms      | 488x    |
| 1          | 0.35ms     | 99.9ms      | 290x    |
| 2          | 0.41ms     | 149.6ms     | 362x    |
| 3          | 0.65ms     | 201.0ms     | 310x    |
| 5          | 1.02ms     | 300.5ms     | 294x    |
| 10         | 1.86ms     | 549.8ms     | 295x    |

### Delta Parameter (n=10000)

| Delta    | fastlowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| none (0) | 32.31ms    | 454.1ms     | 14x     |
| small    | 0.30ms     | 1.62ms      | 5.5x    |
| medium   | 0.31ms     | 0.87ms      | 2.8x    |
| large    | 0.28ms     | 0.53ms      | 1.9x    |

### Pathological Cases (n=5000)

| Case             | fastlowess | statsmodels | Speedup |
|------------------|------------|-------------|---------|
| clustered        | 0.40ms     | 172.2ms     | 426x    |
| constant_y       | 0.37ms     | 141.2ms     | 380x    |
| extreme_outliers | 1.55ms     | 512.0ms     | 331x    |
| high_noise       | 1.64ms     | 463.3ms     | 283x    |

### Real-World Scenarios

#### Financial Time Series

| Size    | fastlowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.20ms     | 7.3ms       | 36x     |
| 1,000   | 0.13ms     | 15.3ms      | 122x    |
| 5,000   | 0.50ms     | 115.0ms     | 232x    |
| 10,000  | 0.72ms     | 337.8ms     | 472x    |

#### Scientific Measurements

| Size    | fastlowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.15ms     | 10.1ms      | 68x     |
| 1,000   | 0.24ms     | 21.9ms      | 91x     |
| 5,000   | 0.57ms     | 178.4ms     | 311x    |
| 10,000  | 1.21ms     | 522.4ms     | 432x    |

#### Genomic Methylation (with delta=100)

| Size    | fastlowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 0.71ms     | 20.6ms      | 29x     |
| 5,000   | 8.33ms     | 153.4ms     | 18x     |
| 10,000  | 27.63ms    | 457.2ms     | 17x     |
| 50,000  | 668.22ms   | 7214.6ms    | 11x     |

## Notes

- **Parallel Execution**: Enabled via Rust/Rayon.
- Benchmarks use standard Python timing with warmup and 10 iterations.
- Both use identical scenarios with reproducible RNG (seed=42).
- Python package: `fastlowess` v0.3.0.
- Reference: `statsmodels` v0.14.x.
- Test date: 2025-12-25.
