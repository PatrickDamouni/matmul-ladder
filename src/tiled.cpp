#include <algorithm>

#include "matmul.h"

// Rung 3: cache blocking. Work on kBlock x kBlock tiles so the pieces of
// A, B and C in use (3 * 64*64*4 B = 48 KB) stay in L1 while they are reused,
// instead of streaming all of B from L2/DRAM for every row of A.
namespace {
constexpr int kBlock = 64;
}

void matmul_tiled(const float* A, const float* B, float* C, int n) {
    std::fill(C, C + static_cast<long>(n) * n, 0.0f);
    for (int ii = 0; ii < n; ii += kBlock) {
        const int iEnd = std::min(ii + kBlock, n);
        for (int kk = 0; kk < n; kk += kBlock) {
            const int kEnd = std::min(kk + kBlock, n);
            for (int jj = 0; jj < n; jj += kBlock) {
                const int jEnd = std::min(jj + kBlock, n);
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
