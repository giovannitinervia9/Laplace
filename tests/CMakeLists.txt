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
  math/test_beta_deriv.cpp
  math/test_student.cpp
  math/test_owen.cpp
  math/test_bessel.cpp
  math/test_bessel_k.cpp
)
laplace_kokkos_sources(main.cpp ${LAPLACE_TEST_SOURCES})
add_executable(laplace_tests main.cpp ${LAPLACE_TEST_SOURCES})
target_link_libraries(laplace_tests PRIVATE Laplace::laplace laplace_doctest)
# reference values (tests/data/reference) and the shared measurement helpers (benchmarks/)
target_compile_definitions(laplace_tests PRIVATE LAPLACE_REFERENCE_DIR="${PROJECT_SOURCE_DIR}/tests/data/reference")
target_include_directories(laplace_tests PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks)

add_test(NAME laplace_tests COMMAND laplace_tests)
