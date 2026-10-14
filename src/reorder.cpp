#include <algorithm>

#include "matmul.h"

// Rung 2: same arithmetic, loops swapped to i-k-j.
// Now the inner loop streams across a row of B and a row of C, both
// contiguous. That makes it cache-friendly and lets the compiler
// auto-vectorize the j loop.
void matmul_reorder(const float* A, const float* B, float* C, int n) {
    const std::ptrdiff_t ld = n;  // 64-bit row stride (see matmul.h)
    std::fill(C, C + ld * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const float a = A[i * ld + k];
            for (int j = 0; j < n; ++j) {
                C[i * ld + j] += a * B[k * ld + j];
            }
        }
    }
}
