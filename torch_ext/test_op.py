"""Correctness of the PyTorch ops vs torch.matmul, plus torch.library.opcheck.

Run: .venv/bin/python torch_ext/test_op.py
"""

import sys

import torch

import loader

SIZES = [1, 2, 3, 4, 5, 15, 16, 17, 31, 33, 63, 64, 65, 100, 129, 255, 256, 300]


def main() -> int:
    loader.load()
    ops = {
        "mm_neon": torch.ops.matmul_ladder.mm_neon,
        "mm_neon_parallel": torch.ops.matmul_ladder.mm_neon_parallel,
    }
    gen = torch.Generator().manual_seed(42)
    failures = 0

    for name, op in ops.items():
        before = failures
        for n in SIZES:
            a = torch.rand(n, n, generator=gen) * 2 - 1
            b = torch.rand(n, n, generator=gen) * 2 - 1
            # Reference in float64, same idea as tests/test_matmul.cpp.
            want = (a.double() @ b.double()).float()
            try:
                torch.testing.assert_close(op(a, b), want, rtol=1e-4, atol=n * 2e-6)
            except AssertionError as e:
                print(f"FAIL {name} n={n}: {str(e).splitlines()[0]}")
                failures += 1

        # Non-contiguous input (a transposed view) must still be right.
        a = torch.rand(64, 64, generator=gen)
        b = torch.rand(64, 64, generator=gen)
        torch.testing.assert_close(op(a.t(), b), a.t() @ b, rtol=1e-4, atol=1e-4)

        # Schema, fake-tensor and dispatch consistency checks from PyTorch.
        torch.library.opcheck(op.default, (torch.rand(32, 32), torch.rand(32, 32)))
        print(f"{'FAIL' if failures > before else 'ok  '} {name}")

    # Bad inputs must raise, not crash.
    # The oversized cases are stride-0 expand() views, so they cost no memory:
    # n = 2^20 + 1 trips the size limit, and n = 2^31 + 1 would wrap negative if
    # the int64 -> int cast were unchecked.
    # Each case must fail with *our* check's message -- an allocation failure
    # deeper in PyTorch would also raise RuntimeError and hide a missing check.
    bad_inputs = [
        (torch.rand(4, 5), "square"),
        (torch.rand(4, 4, dtype=torch.float64), "float32"),
        (torch.zeros(1).expand(2**20 + 1, 2**20 + 1), "supported maximum"),
        (torch.zeros(1).expand(2**31 + 1, 2**31 + 1), "supported maximum"),
    ]
    for name, op in ops.items():
        for bad, expected in bad_inputs:
            try:
                op(bad, bad)
                print(f"FAIL: {name} accepted bad input {tuple(bad.shape)} {bad.dtype}")
                failures += 1
            except RuntimeError as e:
                if expected not in str(e):
                    print(f"FAIL: {name} {tuple(bad.shape)}: wrong error: {str(e).splitlines()[0]}")
                    failures += 1

    print("\nall PyTorch op checks passed" if not failures else f"\n{failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
