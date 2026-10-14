// Benchmarks every rung, plus Apple Accelerate (cblas_sgemm) as a ceiling.
//
// Method: fixed-seed inputs, 1 untimed warmup run, then the median of several
// timed runs. Every result is checked against Accelerate before it is
// reported, so a fast-but-wrong kernel cannot sneak into the table.
//
// Usage: bench [--quick]   (--quick = sizes 256 and 512 only)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <omp.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "matmul.h"

namespace {

using Clock = std::chrono::steady_clock;

void accelerate_sgemm(const float* A, const float* B, float* C, int n) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                1.0f, A, n, B, n, 0.0f, C, n);
}

std::string cpu_name() {
    char buf[256] = {};
    size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) != 0) return "unknown";
    return buf;
}

double time_once(MatmulFn fn, const float* A, const float* B, float* C, int n) {
    const auto t0 = Clock::now();
    fn(A, B, C, n);
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Median of `reps` timed runs after one warmup. Slow kernels get fewer reps.
double median_seconds(MatmulFn fn, const float* A, const float* B, float* C, int n) {
    const double warm = time_once(fn, A, B, C, n);
    const int reps = warm > 1.0 ? 3 : (warm > 0.1 ? 5 : 11);
    std::vector<double> t(reps);
    for (double& x : t) x = time_once(fn, A, B, C, n);
    std::sort(t.begin(), t.end());
    return t[reps / 2];
}

double max_rel_err(const std::vector<float>& got, const std::vector<float>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        num = std::max(num, static_cast<double>(std::fabs(got[i] - want[i])));
        den = std::max(den, static_cast<double>(std::fabs(want[i])));
    }
    return den > 0 ? num / den : num;
}

double gflops(int n, double seconds) {
    return 2.0 * n * n * static_cast<double>(n) / seconds / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const std::vector<int> sizes = quick ? std::vector<int>{256, 512}
                                         : std::vector<int>{256, 512, 1024, 2048};

    std::printf("CPU: %s | OpenMP max threads: %d\n", cpu_name().c_str(), omp_get_max_threads());
    std::printf("Compiler: clang %s\n\n", __clang_version__);

    FILE* csv = std::fopen("results/latest.csv", "w");
    if (csv) std::fprintf(csv, "kernel,n,seconds,gflops,speedup_vs_naive\n");

    std::printf("| kernel | n | median ms | GFLOP/s | speedup vs naive |\n");
    std::printf("|---|---:|---:|---:|---:|\n");

    for (int n : sizes) {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> A(static_cast<size_t>(n) * n), B(A.size()), C(A.size()), ref(A.size());
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);
        accelerate_sgemm(A.data(), B.data(), ref.data(), n);

        double naiveSec = 0.0;
        auto report = [&](const char* name, MatmulFn fn) {
            const double sec = median_seconds(fn, A.data(), B.data(), C.data(), n);
            const double err = max_rel_err(C, ref);
            if (err > 1e-4) {
                std::fprintf(stderr, "WRONG RESULT: %s n=%d max rel err %.3g\n", name, n, err);
                std::exit(1);
            }
            if (naiveSec == 0.0) naiveSec = sec;
            const double speedup = naiveSec / sec;
            std::printf("| %s | %d | %.2f | %.1f | %.1fx |\n", name, n, sec * 1e3, gflops(n, sec), speedup);
            std::fflush(stdout);
            if (csv) std::fprintf(csv, "%s,%d,%.6f,%.3f,%.3f\n", name, n, sec, gflops(n, sec), speedup);
        };

        for (const Kernel& k : kKernels) report(k.name, k.fn);
        report("ref-accelerate", accelerate_sgemm);
    }

    // Thread scaling for rung 5 at the largest size.
    const int n = sizes.back();
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(static_cast<size_t>(n) * n), B(A.size()), C(A.size());
    for (float& x : A) x = dist(rng);
    for (float& x : B) x = dist(rng);

    std::printf("\nThread scaling, 5-omp, n=%d\n\n", n);
    std::printf("| threads | GFLOP/s | speedup vs 1 thread |\n|---:|---:|---:|\n");
    const int maxThreads = omp_get_max_threads();
    double oneThread = 0.0;
    for (int t = 1; t <= maxThreads; t = (t < 8 ? t * 2 : t + 4)) {
        omp_set_num_threads(t);
        const double sec = median_seconds(matmul_omp, A.data(), B.data(), C.data(), n);
        if (t == 1) oneThread = sec;
        std::printf("| %d | %.1f | %.2fx |\n", t, gflops(n, sec), oneThread / sec);
        std::fflush(stdout);
    }

    if (csv) std::fclose(csv);
    return 0;
}
