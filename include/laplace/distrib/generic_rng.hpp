// Random draws by inversion of the quantile (laplace::distrib; note/distrib.md
// section 4), inside Kokkos kernels.
//
//   rng_by_inversion(d, gen)     distrib_quantile(d, u), u uniform on (0, 1)
//   rng_by_inversion(d, seed, index)  the same with the default source,
//                                counter_uniform_source{seed, index}
//   open_unit_uniform(gen)       u = (n + 1/2) 2^-53, n the top 53 bits of gen.urand64()
//   counter_uniform_source       a source keyed by (seed, index): SplitMix64
//
// gen is any object with urand64(): a state of a Kokkos::Random_XorShift64_Pool
// or Random_XorShift1024_Pool (pool.get_state() ... pool.free_state(gen)), or a
// counter_uniform_source. u lies in [2^-54, 1 - 2^-54], symmetric about 1/2, so
// that the quantile is never asked for p = 0 or 1; above 1/2 the routes of
// generic_quantile.hpp work on 1 - u, which is exact there.
//
// counter_uniform_source{seed, index} draws the i-th number of the stream of
// observation `index` as SplitMix64's output at the counter seed + (index 2^32 +
// i) gamma (Steele, Lea and Flood, 2014; gamma = 0x9e3779b97f4a7c15): the draws of
// an observation depend only on (seed, index, i), not on the thread or the
// backend that runs it.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_quantile.hpp>

#include <cstdint>

namespace laplace::distrib {

/**
 * \ingroup distrib_generic
 * \brief Counter-based uniform pseudorandom generator keyed by seed and observation index.
 *
 * Implements the SplitMix64 generator (Steele, Lea and Flood, 2014) evaluating the sequence at counter
 * \f$\mathrm{seed} + ((\mathrm{index} \cdot 2^{32}) + i) \gamma\f$ with \f$\gamma = \mathtt{0x9e3779b97f4a7c15}\f$.
 * Each invocation of `urand64()` increments the internal counter \f$i\f$. Draws for observation `index`
 * depend solely on `(seed, index, i)`, so the numbers are the same on every thread and backend.
 *
 * \see open_unit_uniform(), rng_by_inversion()
 */
struct counter_uniform_source {
  std::uint64_t seed, index;
  std::uint64_t i = 0;
  LAPLACE_HD std::uint64_t urand64() {
    std::uint64_t z = seed + ((index << 32) + i++) * 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
};

/**
 * \ingroup distrib_generic
 * \brief Generates an open unit interval uniform variate \f$u \in [2^{-54}, 1 - 2^{-54}]\f$.
 *
 * Extracts the top 53 bits \f$n\f$ of a 64-bit unsigned integer from `gen.urand64()` and computes:
 * \f[
 * u = \left(n + \frac{1}{2}\right) 2^{-53}
 * \f]
 * The variate is symmetric about \f$1/2\f$ and never 0 or 1, so a quantile is never asked for
 * \f$p = 0\f$ or \f$p = 1\f$.
 *
 * \tparam Gen Generator type providing `std::uint64_t urand64()`.
 * \param gen Generator instance.
 * \return A pseudorandom double in \f$[2^{-54}, 1 - 2^{-54}]\f$.
 *
 * \see counter_uniform_source, rng_by_inversion()
 */
template <class Gen>
LAPLACE_HD_FORCEINLINE double open_unit_uniform(Gen& gen) {
  return (double(gen.urand64() >> 11) + 0.5) * 0x1p-53;
}

/**
 * \ingroup distrib_generic
 * \brief Random variate from a distribution by quantile inversion using an explicit generator.
 *
 * Evaluates \f$Y = F^{-1}(u)\f$ by calling distrib_quantile() on a uniform draw produced by
 * open_unit_uniform(gen).
 *
 * \tparam D Distribution type.
 * \tparam Gen Generator type providing `std::uint64_t urand64()`.
 * \param d Distribution object.
 * \param gen Generator instance (e.g. counter_uniform_source or a Kokkos random pool state).
 * \return A random draw from distribution `d`.
 *
 * \see open_unit_uniform(), distrib_quantile(), counter_uniform_source
 */
template <class D, class Gen>
LAPLACE_HD double rng_by_inversion(const D& d, Gen& gen) {
  return distrib_quantile(d, open_unit_uniform(gen));
}

// The default source (decided 2026-11-12): the draw of observation `index` under
// `seed`, from counter_uniform_source{seed, index}.
/**
 * \ingroup distrib_generic
 * \brief Random variate from a distribution by quantile inversion using the default counter source.
 *
 * Constructs a counter_uniform_source `{seed, index}` and evaluates distrib_quantile() on
 * open_unit_uniform(). The uniform number of observation `index` is the same on every thread and
 * backend.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param seed Global random seed.
 * \param index Observation index.
 * \return A random draw from distribution `d`.
 *
 * \par Speed
 * On gamma1(2, 0.3): 603 ns a call on one gcc thread, 81 on OpenMP (8 threads); 446 on one MSVC
 * host thread and 62 on CUDA (RTX 4070 SUPER).
 *
 * \see counter_uniform_source, distrib_quantile()
 */
template <class D>
LAPLACE_HD double rng_by_inversion(const D& d, std::uint64_t seed, std::uint64_t index) {
  counter_uniform_source src{seed, index};
  return rng_by_inversion(d, src);
}

}  // namespace laplace::distrib
