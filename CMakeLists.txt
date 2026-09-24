cmake_minimum_required(VERSION 3.18...3.31)
project(shufflesnap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SHUFFLESNAP_OPENMP "Parallelize window solves with OpenMP when available" ON)
option(SHUFFLESNAP_NATIVE "Compile for the host CPU (-march=native); off for portable wheels" OFF)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC src/shufflesnap/_core.cpp)

if (SHUFFLESNAP_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP not found: ShuffleSnap will run single-threaded")
  endif()
endif()

if (MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3)
  if (SHUFFLESNAP_NATIVE)
    target_compile_options(_core PRIVATE -march=native)
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION shufflesnap)
