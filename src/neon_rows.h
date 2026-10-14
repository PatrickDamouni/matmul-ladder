#pragma once

// Computes rows [rowBegin, rowEnd) of C = A * B with the NEON register-blocked
// kernel. Shared by rung 4 (one thread does all rows) and rung 5 (threads
// split the rows).
void neon_rows(const float* A, const float* B, float* C, int n, int rowBegin, int rowEnd);
