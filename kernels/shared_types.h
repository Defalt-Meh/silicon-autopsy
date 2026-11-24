// shared_types.h — argument structs shared by the host (C++) and the GPU (MSL).
//
// Both sides include this header, so the memory layout of each struct is
// identical by construction. Only 32-bit unsigned fields are used to avoid
// any alignment/padding ambiguity between the two compilers.
#pragma once

#ifdef __METAL_VERSION__
typedef uint sa_u32;
#else
#include <cstdint>
typedef uint32_t sa_u32;
#endif

// Pointer chase: start at word index `start`, follow `n` dependent loads.
struct ChaseArgs {
    sa_u32 start;
    sa_u32 n;
};

// Bandwidth read: `n_vec` 16-byte vectors, traversed `passes` times by the grid.
struct BwArgs {
    sa_u32 n_vec;
    sa_u32 passes;
};

// ALU clock sentinel: `n` dependent integer LCG steps starting from `seed`.
struct AluArgs {
    sa_u32 seed;
    sa_u32 n;
};
