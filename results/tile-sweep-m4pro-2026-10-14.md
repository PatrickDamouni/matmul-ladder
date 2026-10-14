# Raw `make sweep` output — 2026-10-14 (Apple M4 Pro, clang 21, -O3)


n = 1024 (GFLOP/s, 7 runs)

| variant | min | median | max |
|---|---:|---:|---:|
| 2-reorder (no tiling) | 33.1 | 33.2 | 33.3 |
| tiled i64 k64 j64  (rung 3) | 21.0 | 21.1 | 21.4 |
| tiled i64 k64 j128 | 26.5 | 26.6 | 26.8 |
| tiled i64 k64 j256 | 29.9 | 30.2 | 30.5 |
| tiled i64 k64 j512 | 27.9 | 28.2 | 28.2 |
| tiled i64 k64 j=row | 33.0 | 33.2 | 33.8 |
| tiled i64 k256 j=row | 32.7 | 33.0 | 33.7 |
| tiled i16 k16 j16 | 15.9 | 16.0 | 16.1 |

n = 2048 (GFLOP/s, 7 runs)

| variant | min | median | max |
|---|---:|---:|---:|
| 2-reorder (no tiling) | 30.4 | 32.5 | 32.7 |
| tiled i64 k64 j64  (rung 3) | 19.2 | 19.8 | 19.8 |
| tiled i64 k64 j128 | 22.9 | 23.0 | 23.0 |
| tiled i64 k64 j256 | 27.3 | 27.4 | 27.5 |
| tiled i64 k64 j512 | 27.4 | 27.5 | 27.6 |
| tiled i64 k64 j=row | 32.7 | 32.9 | 33.0 |
| tiled i64 k256 j=row | 32.5 | 32.8 | 32.9 |
| tiled i16 k16 j16 | 12.5 | 12.6 | 12.6 |
