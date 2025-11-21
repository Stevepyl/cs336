cmake_minimum_required(VERSION 4.0)
project(bpe_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(pybind11_DIR .venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Release-ish flags
if (MSVC)
  add_compile_options(/O2 /DNDEBUG)
else()
  add_compile_options(-O3 -DNDEBUG)
endif()

set(OMP_STATIC_LIB "")
set(OMP_INCLUDE_DIR "")

if(APPLE)
  find_program(BREW_PROG brew)
  if(BREW_PROG)
    # Get the prefix where libomp is installed (e.g., /opt/homebrew/opt/libomp)
    execute_process(
      COMMAND ${BREW_PROG} --prefix libomp
      OUTPUT_VARIABLE OMP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(OMP_PREFIX)
      list(APPEND CMAKE_PREFIX_PATH "${OMP_PREFIX}")
      
      # 1. Find the Static Library (.a) explicitly
      find_library(OMP_STATIC_LIB_PATH 
          NAMES libomp.a 
          PATHS "${OMP_PREFIX}/lib" 
          NO_DEFAULT_PATH
      )
      
      # 2. Set the Include Directory explicitly
      set(OMP_INCLUDE_DIR "${OMP_PREFIX}/include")
      
      if(OMP_STATIC_LIB_PATH)
        message(STATUS "Found Static OpenMP Lib: ${OMP_STATIC_LIB_PATH}")
        message(STATUS "Found Static OpenMP Include: ${OMP_INCLUDE_DIR}")
        set(OMP_STATIC_LIB "${OMP_STATIC_LIB_PATH}")
      endif()
    endif()
  endif()
endif()

# Fallback for non-Apple or if static search failed
find_package(OpenMP)

pybind11_add_module(bpe_cpp bpe_cpp/main.cpp)
target_compile_features(bpe_cpp PRIVATE cxx_std_17)

if(OpenMP_CXX_FOUND OR OMP_STATIC_LIB)
  if(APPLE AND OMP_STATIC_LIB)
   # 1. Link against the static .a file
    target_link_libraries(bpe_cpp PRIVATE "${OMP_STATIC_LIB}")
    
    # 2. Add the include directory
    target_include_directories(bpe_cpp PRIVATE "${OMP_INCLUDE_DIR}")
    
    # 3. Force correct flags for Apple Clang
    target_compile_options(bpe_cpp PRIVATE -Xpreprocessor -fopenmp)

    # --- NEW ADDITION: SYMBOL CONFINEMENT ---
    # This tells the linker: "Only export the Python init function. 
    # Keep all OpenMP symbols private/hidden inside this binary."
    target_link_options(bpe_cpp PRIVATE "LINKER:-exported_symbol,_PyInit_bpe_cpp")
  else()
    # Standard dynamic linking for Linux/Windows
    target_link_libraries(bpe_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(bpe_cpp PRIVATE BPECPP_USE_OPENMP=1)
  endif()
endif()

install(TARGETS bpe_cpp DESTINATION .)