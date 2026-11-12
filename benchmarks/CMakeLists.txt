# Measurement tools (accuracy and speed); not part of the test suite.

laplace_kokkos_sources(math_elementary.cpp math_special.cpp math_incomplete.cpp math_ibeta.cpp math_inverse.cpp math_normal.cpp math_gamma_deriv.cpp math_gamma_mixed.cpp math_gamma_ratio.cpp math_beta_deriv.cpp math_student.cpp math_owen.cpp math_bessel.cpp math_bessel_k.cpp math_faa.cpp math_quad.cpp math_dd.cpp math_rel.cpp math_stirling.cpp bijector_scalar.cpp distrib_generic.cpp pdmat_pilot.cpp)

add_executable(math_elementary math_elementary.cpp)
target_link_libraries(math_elementary PRIVATE Laplace::laplace)

add_executable(math_special math_special.cpp)
target_link_libraries(math_special PRIVATE Laplace::laplace)

add_executable(math_incomplete math_incomplete.cpp)
target_link_libraries(math_incomplete PRIVATE Laplace::laplace)

add_executable(math_ibeta math_ibeta.cpp)
target_link_libraries(math_ibeta PRIVATE Laplace::laplace)

add_executable(math_inverse math_inverse.cpp)
target_link_libraries(math_inverse PRIVATE Laplace::laplace)

add_executable(math_normal math_normal.cpp)
target_link_libraries(math_normal PRIVATE Laplace::laplace)

add_executable(math_gamma_deriv math_gamma_deriv.cpp)
target_link_libraries(math_gamma_deriv PRIVATE Laplace::laplace)

add_executable(math_gamma_mixed math_gamma_mixed.cpp)
target_link_libraries(math_gamma_mixed PRIVATE Laplace::laplace)
add_executable(math_gamma_ratio math_gamma_ratio.cpp)
target_link_libraries(math_gamma_ratio PRIVATE Laplace::laplace)

add_executable(math_beta_deriv math_beta_deriv.cpp)
target_link_libraries(math_beta_deriv PRIVATE Laplace::laplace)

add_executable(math_student math_student.cpp)
target_link_libraries(math_student PRIVATE Laplace::laplace)

add_executable(math_owen math_owen.cpp)
target_link_libraries(math_owen PRIVATE Laplace::laplace)

add_executable(math_bessel math_bessel.cpp)
target_link_libraries(math_bessel PRIVATE Laplace::laplace)

add_executable(math_bessel_k math_bessel_k.cpp)
target_link_libraries(math_bessel_k PRIVATE Laplace::laplace)

add_executable(math_faa math_faa.cpp)
target_link_libraries(math_faa PRIVATE Laplace::laplace)

add_executable(math_quad math_quad.cpp)
target_link_libraries(math_quad PRIVATE Laplace::laplace)

add_executable(math_dd math_dd.cpp)
target_link_libraries(math_dd PRIVATE Laplace::laplace)

add_executable(math_rel math_rel.cpp)
target_link_libraries(math_rel PRIVATE Laplace::laplace)

add_executable(bijector_scalar bijector_scalar.cpp)
target_link_libraries(bijector_scalar PRIVATE Laplace::laplace)

add_executable(math_stirling math_stirling.cpp)
target_link_libraries(math_stirling PRIVATE Laplace::laplace)

add_executable(distrib_generic distrib_generic.cpp)
target_link_libraries(distrib_generic PRIVATE Laplace::laplace)

add_executable(pdmat_pilot pdmat_pilot.cpp)
target_link_libraries(pdmat_pilot PRIVATE Laplace::laplace)
