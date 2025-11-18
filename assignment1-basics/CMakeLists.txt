cmake_minimum_required(VERSION 4.0)
project(bpe_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(pybind11_DIR .venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Release-ish flags
if (MSVC)
  add_compile_options(/O2 /DNDEBUG)
else()
  add_compile_options(-O3 -DNDEBUG)
endif()

# Optional OpenMP (comment out if not desired)
# find_package(OpenMP)

pybind11_add_module(bpe_cpp bpe_cpp/main.cpp)
target_compile_features(bpe_cpp PRIVATE cxx_std_17)

# if(OpenMP_CXX_FOUND)
#   target_link_libraries(bpe_cpp PRIVATE OpenMP::OpenMP_CXX)
#   target_compile_definitions(bpe_cpp PRIVATE BPECPP_USE_OPENMP=1)
# endif()

# NumPy include (safer across platforms)
# execute_process(
#   COMMAND "${Python_EXECUTABLE}" -c "import numpy, sys; sys.stdout.write(numpy.get_include())"
#   OUTPUT_VARIABLE NumPy_INCLUDE_DIR
# )
# target_include_directories(bpe_cpp PRIVATE "${NumPy_INCLUDE_DIR}")

install(TARGETS bpe_cpp DESTINATION .)