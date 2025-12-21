# Benchmark Interpretation (fastLowess)

## Summary

The `fastLowess` package demonstrates massive performance gains over Python's `statsmodels`, ranging from **12x to over 3800x** speedup. The addition of **parallel execution** (via Rust/Rayon) and optimized algorithm defaults makes it exceptionally well-suited for high-throughput data processing and large-scale datasets.

## Category Comparison

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Scalability**  | 5       | **577.4x**     | 1375.0x      |
| **Pathological** | 4       | **381.6x**     | 373.4x       |
| **Iterations**   | 6       | **438.1x**     | 426.0x       |
| **Fraction**     | 6       | **336.8x**     | 364.9x       |
| **Financial**    | 4       | **242.1x**     | 263.5x       |
| **Scientific**   | 4       | **165.1x**     | 207.5x       |
| **Genomic**      | 4       | **23.1x**      | 22.7x        |
| **Delta**        | 4       | **3.6x**       | 6.0x         |

## Top 10 FastLowess Wins

| Benchmark        | statsmodels | fastLowess | Speedup   |
|------------------|-------------|------------|-----------|
| scale_100000     | 43727.2ms   | 11.5ms     | **3808.9x** |
| scale_50000      | 11159.9ms   | 5.9ms      | **1901.4x** |
| scale_10000      | 663.1ms     | 1.1ms      | **577.4x**  |
| fraction_0.05    | 197.2ms     | 0.4ms      | **556.5x**  |
| financial_10000  | 497.1ms     | 1.0ms      | **518.8x**  |
| iterations_0     | 74.2ms      | 0.2ms      | **492.9x**  |
| clustered        | 267.8ms     | 0.6ms      | **472.9x**  |
| iterations_1     | 148.5ms     | 0.3ms      | **471.5x**  |
| scale_5000       | 229.9ms     | 0.5ms      | **469.0x**  |
| scientific_10000 | 777.2ms     | 1.7ms      | **464.7x**  |

## Regressions

**None identified.** `fastLowess` outperforms `statsmodels` in all matched benchmarks. The parallel implementation ensures that even at extreme scales (100k points), processing remains sub-20ms.

## Detailed Results

### Scalability (1K - 100K points)

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 0.26ms     | 30.4ms      | 118x    |
| 5,000   | 0.49ms     | 229.9ms     | 469x    |
| 10,000  | 1.15ms     | 663.1ms     | 577x    |
| 50,000  | 5.87ms     | 11159.9ms   | 1901x   |
| 100,000 | 11.48ms    | 43727.2ms   | 3809x   |

### Fraction Variations (n=5000)

| Fraction | fastLowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| 0.05     | 0.35ms     | 197.2ms     | 556x    |
| 0.10     | 0.57ms     | 227.9ms     | 403x    |
| 0.20     | 0.75ms     | 297.0ms     | 395x    |
| 0.30     | 1.28ms     | 357.0ms     | 278x    |
| 0.50     | 1.75ms     | 488.4ms     | 279x    |
| 0.67     | 2.17ms     | 601.6ms     | 278x    |

### Robustness Iterations (n=5000)

| Iterations | fastLowess | statsmodels | Speedup |
|------------|------------|-------------|---------|
| 0          | 0.15ms     | 74.2ms      | 493x    |
| 1          | 0.31ms     | 148.5ms     | 472x    |
| 2          | 0.48ms     | 222.8ms     | 461x    |
| 3          | 0.76ms     | 296.5ms     | 388x    |
| 5          | 1.36ms     | 445.1ms     | 327x    |
| 10         | 1.97ms     | 815.6ms     | 415x    |

### Delta Parameter (n=10000)

| Delta    | fastLowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| none (0) | 45.48ms    | 678.2ms     | 15x     |
| small    | 0.54ms     | 2.28ms      | 4.2x    |
| medium   | 0.43ms     | 1.27ms      | 2.9x    |
| large    | 0.41ms     | 0.76ms      | 1.9x    |

### Pathological Cases (n=5000)

| Case             | fastLowess | statsmodels | Speedup |
|------------------|------------|-------------|---------|
| clustered        | 0.57ms     | 267.8ms     | 473x    |
| constant_y       | 0.55ms     | 230.3ms     | 416x    |
| extreme_outliers | 2.45ms     | 852.0ms     | 347x    |
| high_noise       | 2.83ms     | 726.9ms     | 257x    |

### Real-World Scenarios

#### Financial Time Series

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.20ms     | 10.4ms      | 51x     |
| 1,000   | 0.25ms     | 22.2ms      | 87x     |
| 5,000   | 0.43ms     | 170.9ms     | 397x    |
| 10,000  | 0.96ms     | 497.1ms     | 519x    |

#### Scientific Measurements

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.40ms     | 14.1ms      | 35x     |
| 1,000   | 0.53ms     | 31.6ms      | 60x     |
| 5,000   | 0.99ms     | 268.5ms     | 270x    |
| 10,000  | 1.67ms     | 777.2ms     | 465x    |

#### Genomic Methylation (with delta=100)

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 0.92ms     | 29.5ms      | 32x     |
| 5,000   | 9.06ms     | 227.3ms     | 25x     |
| 10,000  | 31.46ms    | 662.8ms     | 21x     |
| 50,000  | 886.84ms   | 11205.2ms   | 13x     |

## Notes

- **Parallel Execution**: Enabled via Rust/Rayon.
- Benchmarks use standard Python timing with warmup and 10 iterations.
- Both use identical scenarios with reproducible RNG (seed=42).
- Python package: `fastLowess` v0.2.0 (running on `fastLowess` v0.2.x rust crate).
- Reference: `statsmodels` v0.14.x.
- Test date: 2025-12-21.
