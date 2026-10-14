#include <algorithm>

#include "matmul.h"

// Rung 2: same arithmetic, loops swapped to i-k-j.
// Now the inner loop streams across a row of B and a row of C, both
// contiguous. That makes it cache-friendly and lets the compiler
// auto-vectorize the j loop.
void matmul_reorder(const float* A, const float* B, float* C, int n) {
    std::fill(C, C + static_cast<long>(n) * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const float a = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += a * B[k * n + j];
            }
        }
    }
}
