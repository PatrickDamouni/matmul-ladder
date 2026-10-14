"""JIT-builds torch_ext/op.cpp + src/neon.cpp and registers the ops.

After `load()`, the kernels are available as
torch.ops.matmul_ladder.mm_neon and torch.ops.matmul_ladder.mm_neon_parallel.
"""

from pathlib import Path

import torch
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parent.parent
_loaded = False


def load(verbose: bool = False) -> None:
    global _loaded
    if _loaded:
        return
    # This torch build uses OpenMP for at::parallel_for (AT_PARALLEL_OPENMP),
    # and parallel_for is a header template that expands *inside our code*.
    # Without -fopenmp its `#pragma omp parallel` is silently ignored and the
    # "parallel" op runs on one thread. Link torch's own bundled libomp so the
    # process still has exactly one OpenMP runtime.
    torch_lib = Path(torch.__file__).parent / "lib"
    cpp_extension.load(
        name="matmul_ladder",
        sources=[str(ROOT / "torch_ext" / "op.cpp"), str(ROOT / "src" / "neon.cpp")],
        extra_cflags=["-O3", f"-I{ROOT / 'include'}", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[f"-L{torch_lib}", "-lomp", f"-Wl,-rpath,{torch_lib}"],
        is_python_module=False,  # ops register via TORCH_LIBRARY, not a Python module
        verbose=verbose,
    )

    # Shape-only "fake" kernels so torch.compile / opcheck can trace the ops
    # without running them.
    for op in ("mm_neon", "mm_neon_parallel"):

        @torch.library.register_fake(f"matmul_ladder::{op}")
        def _fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            torch._check(a.dim() == 2 and a.shape == b.shape and a.shape[0] == a.shape[1])
            return torch.empty_like(a)

    _loaded = True
