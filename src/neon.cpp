#if !defined(__ARM_NEON)
#error "matmul-ladder rung 4 needs an ARM CPU with NEON (e.g. Apple Silicon)"
#endif

#include <arm_neon.h>

#include <algorithm>

#include "matmul.h"
#include "neon_rows.h"

// Rung 4: hand-written NEON micro-kernel with register blocking.
//
// Rungs 2-3 still load and store C on every k step. Here a 4 x 16 block of C
// lives in 16 NEON registers (4 rows x 4 vectors of 4 floats) for a whole
// k-panel, so per k step we do 4 vector loads of B, 4 scalar loads of A and
// 16 fused multiply-adds -- and touch C only once per panel.
namespace {

constexpr int kMR = 4;    // rows of C per micro-kernel
constexpr int kNR = 16;   // columns of C per micro-kernel (4 x float32x4)
constexpr int kKC = 256;  // k-panel depth: B[kk..kk+256][j..j+16] = 16 KB, fits L1

// C[0..4][0..16] += A[0..4][k0..k1] * B[k0..k1][0..16]  (all offsets pre-applied)
inline void micro_4x16(const float* A, const float* B, float* C, int n, int k0, int k1) {
    float32x4_t c[kMR][4];
    for (int r = 0; r < kMR; ++r) {
        for (int v = 0; v < 4; ++v) {
            c[r][v] = vld1q_f32(C + r * n + v * 4);
        }
    }
    for (int k = k0; k < k1; ++k) {
        const float* b = B + static_cast<long>(k) * n;
        const float32x4_t b0 = vld1q_f32(b);
        const float32x4_t b1 = vld1q_f32(b + 4);
        const float32x4_t b2 = vld1q_f32(b + 8);
        const float32x4_t b3 = vld1q_f32(b + 12);
        for (int r = 0; r < kMR; ++r) {
            const float a = A[r * n + k];
            c[r][0] = vfmaq_n_f32(c[r][0], b0, a);
            c[r][1] = vfmaq_n_f32(c[r][1], b1, a);
            c[r][2] = vfmaq_n_f32(c[r][2], b2, a);
            c[r][3] = vfmaq_n_f32(c[r][3], b3, a);
        }
    }
    for (int r = 0; r < kMR; ++r) {
        for (int v = 0; v < 4; ++v) {
            vst1q_f32(C + r * n + v * 4, c[r][v]);
        }
    }
}

// Scalar fallback for the ragged edges when n is not a multiple of 4 / 16.
void scalar_block(const float* A, const float* B, float* C, int n,
                  int i0, int i1, int j0, int j1, int k0, int k1) {
    for (int i = i0; i < i1; ++i) {
        for (int k = k0; k < k1; ++k) {
            const float a = A[i * n + k];
            for (int j = j0; j < j1; ++j) {
                C[i * n + j] += a * B[k * n + j];
            }
        }
    }
}

}  // namespace

void neon_rows(const float* A, const float* B, float* C, int n, int rowBegin, int rowEnd) {
    std::fill(C + static_cast<long>(rowBegin) * n, C + static_cast<long>(rowEnd) * n, 0.0f);

    const int rowsFull = rowBegin + (rowEnd - rowBegin) / kMR * kMR;
    const int colsFull = n / kNR * kNR;

    for (int kk = 0; kk < n; kk += kKC) {
        const int kEnd = std::min(kk + kKC, n);
        for (int i = rowBegin; i < rowsFull; i += kMR) {
            for (int j = 0; j < colsFull; j += kNR) {
                micro_4x16(A + i * n, B + j, C + i * n + j, n, kk, kEnd);
            }
            scalar_block(A, B, C, n, i, i + kMR, colsFull, n, kk, kEnd);
        }
        scalar_block(A, B, C, n, rowsFull, rowEnd, 0, n, kk, kEnd);
    }
}

void matmul_neon(const float* A, const float* B, float* C, int n) {
    neon_rows(A, B, C, n, 0, n);
}
