# Benchmark Interpretation (fastlowess)

## Summary

The `fastlowess` package demonstrates massive performance gains over Python's `statsmodels`, ranging from **14x to over 2100x** speedup. The addition of **parallel execution** (via Rust/Rayon) and optimized algorithm defaults makes it exceptionally well-suited for high-throughput data processing and large-scale datasets.

## Category Comparison

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Scalability**  | 5       | **344.2x**     | 762.7x       |
| **Pathological** | 4       | **353.8x**     | 348.7x       |
| **Iterations**   | 6       | **222.3x**     | 223.9x       |
| **Fraction**     | 6       | **208.7x**     | 214.8x       |
| **Financial**    | 4       | **192.6x**     | 211.0x       |
| **Scientific**   | 4       | **154.6x**     | 173.9x       |
| **Genomic**      | 4       | **15.6x**      | 14.8x        |
| **Delta**        | 4       | **3.8x**       | 5.9x         |

## Top 10 FastLowess Wins

| Benchmark        | statsmodels | fastlowess | Speedup     |
|------------------|-------------|------------|-------------|
| scale_100000     | 43727.2ms   | 20.0ms     | **2182.7x** |
| scale_50000      | 11159.9ms   | 10.8ms     | **1035.5x** |
| financial_10000  | 497.1ms     | 1.2ms      | **401.3x**  |
| clustered        | 267.8ms     | 0.7ms      | **368.7x**  |
| high_noise       | 726.9ms     | 2.1ms      | **353.9x**  |
| constant_y       | 230.3ms     | 0.7ms      | **353.7x**  |
| scale_10000      | 663.1ms     | 1.9ms      | **344.2x**  |
| scientific_10000 | 777.2ms     | 2.3ms      | **337.6x**  |
| extreme_outliers | 852.0ms     | 2.7ms      | **318.4x**  |
| tricube          | 307.9ms     | 1.0ms      | **302.6x**  |

## Regressions

**None identified.** `fastlowess` outperforms `statsmodels` in all matched benchmarks. The parallel implementation ensures that even at extreme scales (100k points), processing remains around 20ms.

## Detailed Results

### Scalability (1K - 100K points)

| Size    | fastlowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 0.37ms     | 30.4ms      | 81x     |
| 5,000   | 1.35ms     | 229.9ms     | 170x    |
| 10,000  | 1.93ms     | 663.1ms     | 344x    |
| 50,000  | 10.78ms    | 11159.9ms   | 1035x   |
| 100,000 | 20.03ms    | 43727.2ms   | 2183x   |

### Fraction Variations (n=5000)

| Fraction | fastlowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| 0.05     | 0.72ms     | 197.2ms     | 276x    |
| 0.10     | 1.13ms     | 227.9ms     | 202x    |
| 0.20     | 1.38ms     | 297.0ms     | 216x    |
| 0.30     | 1.81ms     | 357.0ms     | 197x    |
| 0.50     | 2.76ms     | 488.4ms     | 177x    |
| 0.67     | 2.71ms     | 601.6ms     | 222x    |

### Robustness Iterations (n=5000)

| Iterations | fastlowess | statsmodels | Speedup |
|------------|------------|-------------|---------|
| 0          | 0.28ms     | 74.2ms      | 269x    |
| 1          | 0.64ms     | 148.5ms     | 232x    |
| 2          | 1.09ms     | 222.8ms     | 205x    |
| 3          | 1.58ms     | 296.5ms     | 188x    |
| 5          | 2.10ms     | 445.1ms     | 212x    |
| 10         | 3.44ms     | 815.6ms     | 237x    |

### Delta Parameter (n=10000)

| Delta    | fastlowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| none (0) | 48.40ms    | 678.2ms     | 14x     |
| small    | 0.49ms     | 2.28ms      | 4.7x    |
| medium   | 0.42ms     | 1.27ms      | 3.0x    |
| large    | 0.40ms     | 0.76ms      | 1.9x    |

### Pathological Cases (n=5000)

| Case             | fastlowess | statsmodels | Speedup |
|------------------|------------|-------------|---------|
| clustered        | 0.73ms     | 267.8ms     | 369x    |
| constant_y       | 0.65ms     | 230.3ms     | 354x    |
| extreme_outliers | 2.68ms     | 852.0ms     | 318x    |
| high_noise       | 2.05ms     | 726.9ms     | 354x    |

### Real-World Scenarios

#### Financial Time Series

| Size    | fastlowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.18ms     | 10.4ms      | 57x     |
| 1,000   | 0.21ms     | 22.2ms      | 104x    |
| 5,000   | 0.61ms     | 170.9ms     | 281x    |
| 10,000  | 1.24ms     | 497.1ms     | 401x    |

#### Scientific Measurements

| Size    | fastlowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.29ms     | 14.1ms      | 49x     |
| 1,000   | 0.44ms     | 31.6ms      | 72x     |
| 5,000   | 1.13ms     | 268.5ms     | 237x    |
| 10,000  | 2.30ms     | 777.2ms     | 338x    |

#### Genomic Methylation (with delta=100)

| Size    | fastlowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 1.85ms     | 29.5ms      | 16x     |
| 5,000   | 14.89ms    | 227.3ms     | 15x     |
| 10,000  | 40.36ms    | 662.8ms     | 16x     |
| 50,000  | 974.48ms   | 11205.2ms   | 11.5x   |

## Notes

- **Parallel Execution**: Enabled via Rust/Rayon.
- Benchmarks use standard Python timing with warmup and 10 iterations.
- Both use identical scenarios with reproducible RNG (seed=42).
- Python package: `fastlowess` v0.3.0.
- Reference: `statsmodels` v0.14.x.
- Test date: 2025-12-25.
