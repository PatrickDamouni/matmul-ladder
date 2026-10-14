// Registers the NEON rung as PyTorch custom operators:
//   torch.ops.matmul_ladder.mm_neon(a, b)           -- rung 4, one thread
//   torch.ops.matmul_ladder.mm_neon_parallel(a, b)  -- rung 4 split over
//       PyTorch's own thread pool (at::parallel_for), not Homebrew libomp,
//       so we never load a second OpenMP runtime into the torch process.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>

#include "matmul.h"
#include "../src/neon_rows.h"

namespace matmul_ladder {

namespace {

constexpr int64_t kRowsPerTask = 32;  // same split as rung 5

void check_inputs(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "mm_neon: tensors must be on CPU");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
                "mm_neon: tensors must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "mm_neon: tensors must be 2-D");
    TORCH_CHECK(a.size(0) == a.size(1) && a.sizes() == b.sizes(),
                "mm_neon: only square n x n inputs of equal size are supported, got ",
                a.sizes(), " and ", b.sizes());
    // The kernels take n as int; refuse anything past the documented contract
    // (matmul.h) instead of letting the int64 -> int cast below wrap around.
    TORCH_CHECK(a.size(0) <= kMaxN, "mm_neon: n = ", a.size(0),
                " exceeds the supported maximum of ", kMaxN);
}

}  // namespace

at::Tensor mm_neon(const at::Tensor& a, const at::Tensor& b) {
    check_inputs(a, b);
    const at::Tensor ac = a.contiguous();
    const at::Tensor bc = b.contiguous();
    at::Tensor c = at::empty_like(ac);
    matmul_neon(ac.data_ptr<float>(), bc.data_ptr<float>(), c.data_ptr<float>(),
                static_cast<int>(ac.size(0)));
    return c;
}

at::Tensor mm_neon_parallel(const at::Tensor& a, const at::Tensor& b) {
    check_inputs(a, b);
    const at::Tensor ac = a.contiguous();
    const at::Tensor bc = b.contiguous();
    at::Tensor c = at::empty_like(ac);
    const int n = static_cast<int>(ac.size(0));
    const float* A = ac.data_ptr<float>();
    const float* B = bc.data_ptr<float>();
    float* C = c.data_ptr<float>();
    const int64_t tasks = (n + kRowsPerTask - 1) / kRowsPerTask;
    at::parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t t = begin; t < end; ++t) {
            const int r0 = static_cast<int>(t * kRowsPerTask);
            neon_rows(A, B, C, n, r0, std::min(r0 + static_cast<int>(kRowsPerTask), n));
        }
    });
    return c;
}

TORCH_LIBRARY(matmul_ladder, m) {
    m.def("mm_neon(Tensor a, Tensor b) -> Tensor");
    m.def("mm_neon_parallel(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(matmul_ladder, CPU, m) {
    m.impl("mm_neon", &mm_neon);
    m.impl("mm_neon_parallel", &mm_neon_parallel);
}

}  // namespace matmul_ladder
