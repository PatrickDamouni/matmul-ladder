// Why does rung 3 (64x64x64 tiles) lose to rung 2 (plain i-k-j)?
//
// Hypothesis: a 64-wide j tile makes the auto-vectorized inner loop too short
// (64 floats = 4 iterations of a 4-wide x 4-interleaved vector body), so loop
// overhead dominates -- and the M4's large caches leave tiling nothing to fix.
// Test: hold everything else fixed and widen only the j tile, then the k tile.
//
// Reports min / median / max GFLOP/s over 7 timed runs after 1 warmup.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "matmul.h"
#include "../src/tiled_impl.h"

namespace {

struct Variant {
    const char* name;
    MatmulFn fn;
};

const Variant kVariants[] = {
    {"2-reorder (no tiling)", matmul_reorder},
    {"tiled i64 k64 j64  (rung 3)", tiled_impl<64, 64, 64>},
    {"tiled i64 k64 j128", tiled_impl<64, 64, 128>},
    {"tiled i64 k64 j256", tiled_impl<64, 64, 256>},
    {"tiled i64 k64 j512", tiled_impl<64, 64, 512>},
    {"tiled i64 k64 j=row", tiled_impl<64, 64, 0>},
    {"tiled i64 k256 j=row", tiled_impl<64, 256, 0>},
    {"tiled i16 k16 j16", tiled_impl<16, 16, 16>},
};

double gflops(int n, double sec) { return 2.0 * n * n * static_cast<double>(n) / sec / 1e9; }

}  // namespace

int main() {
    constexpr int kReps = 7;
    for (int n : {1024, 2048}) {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> A(static_cast<size_t>(n) * n), B(A.size()), C(A.size());
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);

        std::printf("\nn = %d (GFLOP/s, %d runs)\n\n", n, kReps);
        std::printf("| variant | min | median | max |\n|---|---:|---:|---:|\n");
        for (const Variant& v : kVariants) {
            v.fn(A.data(), B.data(), C.data(), n);  // warmup
            std::vector<double> g(kReps);
            for (double& x : g) {
                const auto t0 = std::chrono::steady_clock::now();
                v.fn(A.data(), B.data(), C.data(), n);
                x = gflops(n, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(g.begin(), g.end());
            std::printf("| %s | %.1f | %.1f | %.1f |\n", v.name, g.front(), g[kReps / 2], g.back());
            std::fflush(stdout);
        }
    }
    return 0;
}
