# Force llama.cpp's custom static runtime toggle
set(LLAMA_STATIC_CRUNTIME ON CACHE BOOL "Force static CRT" FORCE)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "Force static CRT" FORCE)