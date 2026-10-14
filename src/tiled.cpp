#include "matmul.h"
#include "tiled_impl.h"

// Rung 3: cache blocking. Work on 64 x 64 tiles so the pieces of A, B and C
// in use (3 * 64*64*4 B = 48 KB) stay in L1 while they are reused, instead of
// streaming all of B from L2/DRAM for every row of A.
//
// On the M4 Pro this is *slower* than rung 2 -- see bench/tile_sweep.cpp and
// the README for why.
void matmul_tiled(const float* A, const float* B, float* C, int n) {
    tiled_impl<64, 64, 64>(A, B, C, n);
}
