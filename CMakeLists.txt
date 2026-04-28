cmake_minimum_required(VERSION 3.10)
project(RotorQuant VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Core library (object library for reuse) ──────────────────────
add_library(rotorquant_core OBJECT
    rotorQuant.cpp
    rotation.cpp
    lloyd_max.cpp
)
target_include_directories(rotorquant_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# ── OpenMP (optional, for parallel encode/decode) ────────────────
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rotorquant_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — parallel paths enabled")
endif()

# ── C++ test executable ──────────────────────────────────────────
add_executable(test_rotor_quant test_rotorQuant.cpp)
target_link_libraries(test_rotor_quant PRIVATE rotorquant_core)

# ── pybind11 module (optional) ───────────────────────────────────
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(rotorquant bindings.cpp)
    target_link_libraries(rotorquant PRIVATE rotorquant_core)
    message(STATUS "pybind11 found — Python module target 'rotorquant' available")
else()
    message(STATUS "pybind11 not found — skipping Python module (install with: pip install pybind11)")
endif()
