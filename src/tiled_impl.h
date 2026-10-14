#pragma once

#include <algorithm>

// Cache-blocked i-k-j matmul with compile-time block sizes. Rung 3 uses
// <64, 64, 64>; bench/tile_sweep.cpp instantiates other shapes to find out
// why that rung loses to plain loop reordering. BJ <= 0 means "full row".
template <int BI, int BK, int BJ>
void tiled_impl(const float* A, const float* B, float* C, int n) {
    const int bj = BJ > 0 ? BJ : n;
    std::fill(C, C + static_cast<long>(n) * n, 0.0f);
    for (int ii = 0; ii < n; ii += BI) {
        const int iEnd = std::min(ii + BI, n);
        for (int kk = 0; kk < n; kk += BK) {
            const int kEnd = std::min(kk + BK, n);
            for (int jj = 0; jj < n; jj += bj) {
                const int jEnd = std::min(jj + bj, n);
                for (int i = ii; i < iEnd; ++i) {
                    for (int k = kk; k < kEnd; ++k) {
                        const float a = A[i * n + k];
                        for (int j = jj; j < jEnd; ++j) {
                            C[i * n + j] += a * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}
