#pragma once

#include <cstddef>

// Every kernel computes C = A * B for square, row-major n x n float matrices.
// C is fully overwritten; A and B are read-only.
//
// Contract: 0 <= n <= kMaxN. Element offsets like i * n are computed in 64-bit
// (std::ptrdiff_t) because i * n overflows a 32-bit int once n >= 46341.
// kMaxN (4 TB per matrix) is far past any real machine; it exists so that
// the int loop bounds (e.g. jj + blockSize) can never overflow either.
inline constexpr int kMaxN = 1 << 20;

using MatmulFn = void (*)(const float* A, const float* B, float* C, int n);

void matmul_naive(const float* A, const float* B, float* C, int n);
void matmul_reorder(const float* A, const float* B, float* C, int n);
void matmul_tiled(const float* A, const float* B, float* C, int n);
void matmul_neon(const float* A, const float* B, float* C, int n);
void matmul_omp(const float* A, const float* B, float* C, int n);

struct Kernel {
    const char* name;
    MatmulFn fn;
};

// The ladder, in order. Each rung builds on the one before it.
inline constexpr Kernel kKernels[] = {
    {"1-naive", matmul_naive},
    {"2-reorder", matmul_reorder},
    {"3-tiled", matmul_tiled},
    {"4-neon", matmul_neon},
    {"5-omp", matmul_omp},
};
