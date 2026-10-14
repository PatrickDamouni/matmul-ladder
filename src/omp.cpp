#include <omp.h>

#include <algorithm>

#include "matmul.h"
#include "neon_rows.h"

// Rung 5: rung 4 on every core. Rows of C are independent, so threads split
// them with no locking. schedule(dynamic) matters on Apple Silicon: the
// efficiency cores are slower, and a static split would leave the
// performance cores waiting on them.
namespace {
constexpr int kRowsPerTask = 32;  // multiple of the 4-row micro-kernel
}

void matmul_omp(const float* A, const float* B, float* C, int n) {
    const int tasks = (n + kRowsPerTask - 1) / kRowsPerTask;
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < tasks; ++t) {
        const int r0 = t * kRowsPerTask;
        neon_rows(A, B, C, n, r0, std::min(r0 + kRowsPerTask, n));
    }
}
