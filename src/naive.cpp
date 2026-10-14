#include "matmul.h"

// Rung 1: the textbook triple loop (i-j-k).
// The inner loop walks B down a column: B[k*n + j] jumps n floats per step,
// so almost every load of B touches a new cache line once n is large.
void matmul_naive(const float* A, const float* B, float* C, int n) {
    const std::ptrdiff_t ld = n;  // 64-bit row stride (see matmul.h)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * ld + k] * B[k * ld + j];
            }
            C[i * ld + j] = sum;
        }
    }
}
