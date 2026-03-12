cmake_minimum_required(VERSION 3.15)
project(quakfilter LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Find Python
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Detect Nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NB_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Eigen — default: fetch Eigen 5.0.0 via FetchContent (requires network).
# Set QUAK_USE_SYSTEM_EIGEN=ON to use a locally-installed Eigen 3.x instead.
option(QUAK_USE_SYSTEM_EIGEN "Use system-installed Eigen instead of fetching Eigen 5.0.0" OFF)

if(QUAK_USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.3 REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

# OpenMP — optional, enables parallel predict in TrackPool
find_package(OpenMP QUIET)

# Build the C++ extension module as quak._core
nanobind_add_module(_core NOMINSIZE csrc/bindings.cpp)

target_include_directories(_core PRIVATE csrc/include csrc)

target_link_libraries(_core PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core PRIVATE QUAK_HAS_OPENMP)
endif()

# Optimization flags
if(MSVC)
    target_compile_options(_core PRIVATE /O2)
    if(OpenMP_CXX_FOUND)
        target_compile_options(_core PRIVATE /openmp)
    endif()
else()
    target_compile_options(_core PRIVATE -O3)
endif()

# Install _core.so into the quak package directory
install(TARGETS _core LIBRARY DESTINATION quak)
