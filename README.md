# matmul-ladder

Single-precision matrix multiply (`C = A × B`) in C++, made faster one rung at a time on Apple Silicon. Every rung is measured and checked for correctness.

```
1-naive  →  2-reorder  →  3-tiled  →  4-neon  →  5-omp       (ref: Apple Accelerate)
```

**Headline (M4 Pro, n = 2048):** a hand-written NEON kernel reaches **90 GFLOP/s on one core**, 2.7× the compiler's auto-vectorized loop. Across 12 threads it reaches **626 GFLOP/s**, about **22 % of Apple Accelerate**.

## Quick start

```bash
brew install libomp     # OpenMP runtime; Apple clang doesn't ship one
make test               # correctness: 5 kernels × 18 sizes vs a double-precision reference
make asan               # same tests under AddressSanitizer + UBSan
make bench              # full benchmark (~1 min); `make bench-quick` for n ≤ 512
```

## Results

Apple M4 Pro (8 performance + 4 efficiency cores), Apple clang 21, `-O3`. Each entry is the median of 3–11 timed runs after one warmup run. Every result is checked against Accelerate before it is reported.

| kernel | n=256 | n=512 | n=1024 | n=2048 | speedup vs naive @2048 |
|---|---:|---:|---:|---:|---:|
| 1-naive | 3.3 | 3.1 | 2.6 | 1.7 | 1× |
| 2-reorder | 34.4 | 35.4 | 34.5 | 33.5 | 20× |
| 3-tiled | 30.7 | 21.9 | 22.0 | 20.1 | 12× |
| 4-neon (1 core) | 101.9 | 104.0 | 100.5 | 90.5 | 54× |
| 5-omp (12 threads) | 275.7 | 675.0 | 716.2 | 626.4 | 373× |
| *ref: Accelerate `cblas_sgemm`* | *1778* | *2567* | *1906* | *2823* | — |

Values are GFLOP/s (higher is better), computed as `2·n³ / time`. Raw output is in [`results/m4pro-2026-10-14.md`](results/m4pro-2026-10-14.md).

**Thread scaling (5-omp, n=2048):**

| threads | 1 | 2 | 4 | 8 | 12 |
|---|---:|---:|---:|---:|---:|
| GFLOP/s | 90.5 | 166.2 | 335.7 | 603.0 | 633.5 |
| speedup | 1.00× | 1.84× | 3.71× | 6.66× | 7.00× |

## What each rung changes

1. **naive (i-j-k).** The inner loop walks *down a column* of B, a stride of `n` floats, so nearly every load is a cache miss at large n. Performance gets *worse* as n grows: 3.3 → 1.7 GFLOP/s.
2. **reorder (i-k-j).** Same math, loops swapped. The inner loop now streams across contiguous rows of B and C, and clang auto-vectorizes it. The one-line change gives 10–20×.
3. **tiled.** 64×64 cache blocks. **This made things slower**, not faster. See the open question below.
4. **neon.** A hand-written 4×16 micro-kernel. A 4-row by 16-column block of C stays in 16 NEON registers for a 256-deep k-panel. Each k step does 4 vector loads of B and 16 fused multiply-adds (`vfmaq_n_f32`), and C is touched only once per panel instead of every step.
5. **omp.** Rung 4 split across threads by rows of C (`schedule(dynamic)`, 32-row tasks). Rows are independent, so no locking is needed.

## Open question: why did tiling lose?

The textbook says blocking should help, but on this machine it lost to plain loop reordering at every size. These are hypotheses I haven't verified yet:

- **The tiles shorten the vectorized inner loop** from `n` iterations to 64, which adds loop overhead and prologue/epilogue cost per tile.
- **M4's caches are large** (128 KB L1d per performance core, 16 MB shared L2), so the reorder loop's working set may already fit well enough that blocking has nothing to fix at these sizes.
- **C is still loaded and stored on every k step.** Blocking alone doesn't fix that. Register blocking (rung 4) does.

Next step: profile rungs 2 and 3 with Instruments (CPU Counters) and compare L1/L2 miss rates, then sweep the block size.

## Other things the data shows

- **8 → 12 threads adds only 5 %.** The last 4 threads land on efficiency cores, which are much slower. The 1 → 8 curve (6.66× on 8 performance cores) is the real scaling story.
- **Speedup vs. naive flatters itself at large n**, because naive collapses (1.7 GFLOP/s). The honest single-core comparison is rung 4 vs. rung 2: **2.7×** at n=2048, against what the compiler does on its own.
- **Accelerate is 4.5× faster than rung 5.** It likely uses Apple's dedicated matrix hardware (AMX/SME) rather than NEON, so it's a ceiling to measure against, not a target this approach can reach.

## Correctness

`tests/test_matmul.cpp` checks every kernel against a double-precision reference on 18 sizes (1, 2, 3, 4, 5, 15, 16, 17, 31, 33, 63, 64, 65, 100, 129, 255, 256, 300). Odd sizes are deliberate: they exercise the ragged edges of the 4×16 micro-kernel and the 64-wide tiles. The per-element tolerance is `2 · n · ε · Σ|a·b|`, derived from the float32 rounding bound, not a magic constant. Output buffers are pre-filled with NaN, so a kernel that skips an element fails the test.

The tests were checked by planting a bug (dropping the edge-row path in rung 4), which they caught as 24 failing cases.

## Limitations

- Square matrices only; row-major only.
- The NEON rung requires ARM (Apple Silicon or ARM Linux). The bench links Apple Accelerate, so it is macOS-only.
- B is not packed, and there is no prefetching or k-loop unrolling yet. Those are the obvious next steps toward Accelerate.
- Timings come from a laptop with background load, and no run-to-run variance is reported yet.

## Roadmap

- [ ] Profile rungs 2 vs 3 and resolve the tiling question
- [ ] Pack B panels; try 8×12 and 6×16 micro-kernels
- [ ] Report the spread (min/median/max) alongside the median
- [ ] Port to CUDA (Colab T4): naive → shared-memory tiling → register tiling, profiled with Nsight Compute, compared against cuBLAS

## Layout

```
include/matmul.h      kernel signatures + the ladder table
src/                  one file per rung (neon_rows.h is shared by rungs 4 and 5)
tests/test_matmul.cpp correctness vs double-precision reference
bench/bench.cpp       timing, GFLOP/s, Accelerate check, thread sweep
```
