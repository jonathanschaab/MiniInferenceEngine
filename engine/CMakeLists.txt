cmake_minimum_required(VERSION 3.18)
project(MiniInferenceEngine LANGUAGES CXX CUDA)

# 1. Find the NVIDIA CUDA Toolkit on the machine
find_package(CUDAToolkit REQUIRED)

# 2. Tell CMake to build a Shared Library (.so) instead of an executable
add_library(engine SHARED 
    engine.cpp 
    matrix_math.cu
)

# 3. Ensure the library compiles with position-independent code (required for shared libs)
set_target_properties(engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 4. Link the CUDA runtime so our C++ code can use cudaMemcpy
target_link_libraries(engine PRIVATE CUDA::cudart)