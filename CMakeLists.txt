cmake_minimum_required(VERSION 3.25)
project(Laplace VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos 5: Serial and OpenMP with gcc, CUDA with MSVC + nvcc.
find_package(Kokkos 5 REQUIRED)

# With CUDA-enabled Kokkos, the sources that launch kernels are compiled by
# nvcc; Eigen code stays in translation units of its own, compiled as C++.
if(Kokkos_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # nvcc contracts a * b + c into an FMA by default, host compilers for x86-64
  # do not: with OFF, device results can differ from host results in the last bit.
  option(LAPLACE_CUDA_FMAD "Let nvcc contract multiply-adds into FMA" ON)
  if(NOT LAPLACE_CUDA_FMAD)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  endif()
endif()
function(laplace_kokkos_sources)
  if(Kokkos_ENABLE_CUDA)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
  endif()
endfunction()

# The numerical layers are header-only, so that every per-observation
# function is compiled into the Kokkos kernels that call it, on host and device.
add_library(laplace INTERFACE)
add_library(Laplace::laplace ALIAS laplace)
target_include_directories(laplace INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(laplace INTERFACE Kokkos::kokkos)

# exp and log on the host: Laplace's own (default) or the platform's libm
option(LAPLACE_OWN_ELEMENTARY "Use Laplace's own exp and log on the host" ON)
if(NOT LAPLACE_OWN_ELEMENTARY)
  target_compile_definitions(laplace INTERFACE LAPLACE_PLATFORM_ELEMENTARY)
endif()

# Boost.Math (header-only, Boost Software License) for some special functions,
# on the host and on the GPU. Either a directory holding boost/ is given with
# LAPLACE_BOOST_INCLUDE, or an installed Boost is found with find_package.
set(LAPLACE_BOOST_INCLUDE "" CACHE PATH "Directory holding boost/ (Boost >= 1.90); empty: find_package(Boost)")
if(LAPLACE_BOOST_INCLUDE)
  if(NOT EXISTS "${LAPLACE_BOOST_INCLUDE}/boost/math/special_functions/gamma.hpp")
    message(FATAL_ERROR "Boost.Math not found in LAPLACE_BOOST_INCLUDE=${LAPLACE_BOOST_INCLUDE}")
  endif()
  target_include_directories(laplace SYSTEM INTERFACE ${LAPLACE_BOOST_INCLUDE})
else()
  find_package(Boost 1.90 REQUIRED CONFIG)
  target_link_libraries(laplace INTERFACE Boost::headers)
endif()

option(LAPLACE_BUILD_BENCHMARKS "Build the measurement tools" ${PROJECT_IS_TOP_LEVEL})
if(LAPLACE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

option(LAPLACE_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
if(LAPLACE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
