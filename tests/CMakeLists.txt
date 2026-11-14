# doctest 2.4.11 (MIT), single header in third_party/doctest.
add_library(laplace_doctest INTERFACE)
target_include_directories(laplace_doctest SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party)

# One test executable; sources that launch Kokkos kernels are compiled by nvcc
# in the CUDA build. CTest runs the whole executable; single cases are chosen
# with doctest's own options (laplace_tests -tc="<name>").
set(LAPLACE_TEST_SOURCES
  core/test_config.cpp
  math/test_special.cpp
  math/test_ibeta.cpp
  math/test_inverse.cpp
  math/test_normal.cpp
  math/test_gamma_deriv.cpp
  math/test_gamma_mixed.cpp
  math/test_gamma_ratio.cpp
  math/test_beta_deriv.cpp
  math/test_student.cpp
  math/test_owen.cpp
  math/test_bessel.cpp
  math/test_bessel_k.cpp
  math/test_faa_di_bruno.cpp
  math/test_quadrature.cpp
  math/test_double_double.cpp
  bijector/test_scalar.cpp
  bijector/test_affine.cpp
  bijector/test_shifted.cpp
  bijector/test_bounded.cpp
  bijector/test_power_softplus.cpp
  bijector/test_box_cox.cpp
  bijector/test_yeo_johnson.cpp
  bijector/param_common.cpp
  bijector/test_coupled.cpp
  math/test_exprel.cpp
  math/test_log1prel.cpp
  math/test_stirling.cpp
  distrib/test_pilot.cpp
  distrib/test_generic.cpp
  distrib/test_gaussian2.cpp
  distrib/test_gaussian3.cpp
  distrib/test_cauchy.cpp
  distrib/test_logistic.cpp
  distrib/test_gumbel.cpp
  distrib/test_laplace2.cpp
  distrib/test_conditioning.cpp
  distrib/test_exponential.cpp
  distrib/test_chisq.cpp
  distrib/test_gamma2.cpp
  distrib/test_invgauss.cpp
  distrib/test_lognormal1.cpp
  distrib/test_lognormal2.cpp
  math/test_chain_pair.cpp
  pdmat/test_pilot.cpp
  pdmat/test_unstructured.cpp
  pdmat/test_unit_diagonal_scaled.cpp
)
laplace_kokkos_sources(main.cpp ${LAPLACE_TEST_SOURCES})
add_executable(laplace_tests main.cpp ${LAPLACE_TEST_SOURCES})
target_link_libraries(laplace_tests PRIVATE Laplace::laplace laplace_doctest)
# reference values (tests/data/reference) and the shared measurement helpers (benchmarks/)
target_compile_definitions(laplace_tests PRIVATE LAPLACE_REFERENCE_DIR="${PROJECT_SOURCE_DIR}/tests/data/reference")
target_include_directories(laplace_tests PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks)

# The kernel of the scalar-bijector check, one translation unit per total order
# K = 0..6 (bijector/param_order.cpp compiled seven times), so that nvcc
# compiles the seven parts in parallel: in one unit ptxas took 700 s.
laplace_kokkos_sources(bijector/param_order.cpp)
foreach(k RANGE 0 6)
  add_library(laplace_param_order_${k} OBJECT bijector/param_order.cpp)
  target_link_libraries(laplace_param_order_${k} PRIVATE Laplace::laplace laplace_doctest)
  target_compile_definitions(laplace_param_order_${k} PRIVATE LAPLACE_PARAM_ORDER=${k}
    LAPLACE_REFERENCE_DIR="${PROJECT_SOURCE_DIR}/tests/data/reference")
  target_include_directories(laplace_param_order_${k} PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks)
  target_link_libraries(laplace_tests PRIVATE laplace_param_order_${k})
endforeach()

add_test(NAME laplace_tests COMMAND laplace_tests)
