// Laplace: configuration shared by every layer.
//
// Every per-observation function (special functions, links, families,
// parameter maps, bases, penalties) is callable inside a Kokkos kernel, on the
// host and on the device. Such functions are marked LAPLACE_HD, throw no
// exception, allocate no memory, and use only Laplace's own implementations of
// the special functions (never the host libm or the CUDA built-ins), so that
// the same source gives the same result on every platform.
//
// Functions are templates on the scalar type; only double is validated for
// now.
#pragma once

#include <Kokkos_Macros.hpp>

#define LAPLACE_HD KOKKOS_INLINE_FUNCTION
#define LAPLACE_HD_FORCEINLINE KOKKOS_FORCEINLINE_FUNCTION

// A table in device constant memory. With relocatable device code (-rdc) all
// units form one device program with one 64 KB constant bank, so the table is
// inline: one copy per program. Without it nvcc rejects an inline __constant__
// variable, and every unit has its own static copy.
#if defined(__CUDACC_RDC__)
#define LAPLACE_DEVICE_TABLE inline __constant__
#else
#define LAPLACE_DEVICE_TABLE static __constant__
#endif

// A table in ordinary device global memory, read through the caches. The
// constant bank is one 64 KB resource for the whole device program and it is
// nearly full, so a table of several kilobytes goes here instead. It is also
// the better place for a table whose index differs from thread to thread: the
// constant bank serves one address per cycle and serializes a divergent read,
// while a global read of a few kilobytes is served by the caches.
#if defined(__CUDACC_RDC__)
#define LAPLACE_DEVICE_GLOBAL_TABLE inline __device__ const
#else
#define LAPLACE_DEVICE_GLOBAL_TABLE static __device__ const
#endif

namespace laplace {

// The scalar type validated today.
using real = double;

}  // namespace laplace
