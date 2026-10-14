"""Benchmarks the NEON ops against torch.matmul (CPU) and torch.matmul on MPS.

Uses torch.utils.benchmark.Timer, which handles warmup and picks the number
of runs itself (blocked_autorange). Run: .venv/bin/python torch_ext/bench_op.py
"""

import platform
import re

import torch
from torch.utils.benchmark import Timer

import loader


def gflops(n: int, seconds: float) -> float:
    return 2 * n**3 / seconds / 1e9


def time_cpu(stmt: str, a: torch.Tensor, b: torch.Tensor, threads: int) -> float:
    t = Timer(stmt, globals={"torch": torch, "a": a, "b": b}, num_threads=threads)
    return t.blocked_autorange(min_run_time=1.0).median


def time_mps(a: torch.Tensor, b: torch.Tensor) -> float:
    # MPS launches are asynchronous: synchronize inside the timed statement,
    # or we would only time the kernel launch.
    t = Timer(
        "torch.matmul(a, b); torch.mps.synchronize()",
        globals={"torch": torch, "a": a, "b": b},
    )
    return t.blocked_autorange(min_run_time=1.0).median


def main() -> None:
    loader.load()
    threads = torch.get_num_threads()
    blas = re.search(r"BLAS_INFO=(\w+)", torch.__config__.show())
    print(f"torch {torch.__version__} | {platform.machine()} | BLAS: {blas.group(1) if blas else '?'}"
          f" | CPU threads: {threads} | MPS: {torch.backends.mps.is_available()}\n")

    print("| n | mm_neon (1 thr) | torch.matmul (1 thr) | mm_neon_parallel "
          f"({threads} thr) | torch.matmul ({threads} thr) | torch.matmul MPS |")
    print("|---:|---:|---:|---:|---:|---:|")
    for n in (256, 512, 1024, 2048):
        a, b = torch.rand(n, n) * 2 - 1, torch.rand(n, n) * 2 - 1
        row = [
            time_cpu("torch.ops.matmul_ladder.mm_neon(a, b)", a, b, 1),
            time_cpu("torch.matmul(a, b)", a, b, 1),
            time_cpu("torch.ops.matmul_ladder.mm_neon_parallel(a, b)", a, b, threads),
            time_cpu("torch.matmul(a, b)", a, b, threads),
        ]
        cells = [f"{gflops(n, s):.0f}" for s in row]
        if torch.backends.mps.is_available():
            cells.append(f"{gflops(n, time_mps(a.to('mps'), b.to('mps'))):.0f}")
        else:
            cells.append("n/a")
        print(f"| {n} | " + " | ".join(cells) + " |", flush=True)
    print("\nValues are GFLOP/s (2·n³ / median time).")


if __name__ == "__main__":
    main()
