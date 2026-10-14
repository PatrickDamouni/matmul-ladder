#pragma once

// Every kernel computes C = A * B for square, row-major n x n float matrices.
// C is fully overwritten; A and B are read-only.
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
