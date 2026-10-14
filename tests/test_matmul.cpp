// Checks every rung against a double-precision reference.
//
// Tolerance: for a length-n dot product in float32 the rounding error is
// bounded by about n * eps * sum_k |a_ik * b_kj|. We allow 2x that bound per
// element, so the test is principled rather than a magic 1e-3.
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "matmul.h"

namespace {

std::vector<float> random_matrix(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> m(static_cast<size_t>(n) * n);
    for (float& x : m) x = dist(rng);
    return m;
}

// Returns the number of elements outside tolerance.
int check(const Kernel& kernel, int n, std::mt19937& rng) {
    const std::vector<float> A = random_matrix(n, rng);
    const std::vector<float> B = random_matrix(n, rng);
    std::vector<float> C(static_cast<size_t>(n) * n, std::nanf(""));  // poison: kernels must overwrite
    kernel.fn(A.data(), B.data(), C.data(), n);

    const double eps = std::numeric_limits<float>::epsilon();
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double ref = 0.0, mag = 0.0;
            for (int k = 0; k < n; ++k) {
                const double p = static_cast<double>(A[i * n + k]) * B[k * n + j];
                ref += p;
                mag += std::fabs(p);
            }
            const double got = C[i * n + j];
            const double tol = 2.0 * n * eps * mag + 1e-30;
            if (!(std::fabs(got - ref) <= tol)) {  // also catches NaN
                if (bad == 0) {
                    std::printf("    first mismatch at (%d,%d): got %.8g want %.8g tol %.3g\n",
                                i, j, got, ref, tol);
                }
                ++bad;
            }
        }
    }
    return bad;
}

}  // namespace

int main() {
    // Odd sizes on purpose: they exercise the ragged-edge paths of the
    // 4x16 NEON micro-kernel and the 64-wide tiles.
    const int sizes[] = {1, 2, 3, 4, 5, 15, 16, 17, 31, 33, 63, 64, 65, 100, 129, 255, 256, 300};
    std::mt19937 rng(42);
    int failures = 0;
    for (const Kernel& kernel : kKernels) {
        int kernelFailures = 0;
        for (int n : sizes) {
            const int bad = check(kernel, n, rng);
            if (bad) {
                std::printf("FAIL %-10s n=%-4d %d bad elements\n", kernel.name, n, bad);
                ++kernelFailures;
            }
        }
        std::printf("%s %s\n", kernelFailures ? "FAIL" : "ok  ", kernel.name);
        failures += kernelFailures;
    }
    if (failures) {
        std::printf("\n%d failing (kernel, size) cases\n", failures);
        return 1;
    }
    std::printf("\nall %zu kernels passed on %zu sizes\n",
                sizeof(kKernels) / sizeof(kKernels[0]), sizeof(sizes) / sizeof(sizes[0]));
    return 0;
}
