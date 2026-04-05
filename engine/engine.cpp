#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// 1. Forward declaration so C++ knows the CUDA function exists in the .cu file
void launch_tiled_matmul(const float* d_X, const float* d_W, float* d_Y, int N);

// 2. The main function we will eventually expose to Rust via FFI
extern "C" void run_forward_pass(const float* host_X, const float* host_W, float* host_Y, int N) {
    size_t bytes = N * N * sizeof(float);
    float *d_X, *d_W, *d_Y;

    // STEP 1: Allocate VRAM on the GPU (The "Device")
    // This is the GPU equivalent of standard C++ malloc()
    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_W, bytes);
    cudaMalloc(&d_Y, bytes);

    // STEP 2: The PCIe Transfer! (The bottleneck we discussed earlier)
    // Copy the BatchPayload from standard CPU RAM to the newly allocated VRAM
    cudaMemcpy(d_X, host_X, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, host_W, bytes, cudaMemcpyHostToDevice);

    // STEP 3: Trigger the execution 🚀
    // We call the wrapper that actually invokes the <<<blocks, threads>>> syntax
    launch_tiled_matmul(d_X, d_W, d_Y, N);

    // STEP 4: Wait for the GPU to finish calculating
    cudaDeviceSynchronize();

    // STEP 5: Copy the final answers back over PCIe to the CPU RAM
    cudaMemcpy(host_Y, d_Y, bytes, cudaMemcpyDeviceToHost);

    // STEP 6: Prevent Memory Leaks by freeing the VRAM
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
}