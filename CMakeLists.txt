cmake_minimum_required(VERSION 3.18)
project(shufflesnap LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core src/core.cpp)
target_compile_features(_core PRIVATE cxx_std_17)

install(TARGETS _core LIBRARY DESTINATION shufflesnap)
