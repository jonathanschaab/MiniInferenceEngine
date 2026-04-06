#include <cuda_runtime.h>

// We define our tile size (16x16)
#define TILE_SIZE 16

// ---------------------------------------------------------
// 1. THE KERNEL (Runs on the GPU)
// ---------------------------------------------------------
__global__ void tiled_matmul_kernel(const float* X, const float* W, float* Y, int N) {
    // A. Allocate ultra-fast SRAM (Shared Memory) for this specific thread block
    __shared__ float s_X[TILE_SIZE][TILE_SIZE];
    __shared__ float s_W[TILE_SIZE][TILE_SIZE];

    // B. Identify who this thread is and what coordinates it represents
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float dot_product = 0.0;

    // C. Calculate how many times we need to "slide" the tile across the matrix
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // D. The Outer Loop: Sliding across X and down W
    for (int p = 0; p < num_tiles; ++p) {
        
        // --- COOPERATIVE FETCH ---
        // Each thread grabs exactly ONE number from slow VRAM to load into the SRAM tile.
        // We include boundary checks in case our matrix size isn't a perfect multiple of 16.
        if (row < N && (p * TILE_SIZE + tx) < N) {
            s_X[ty][tx] = X[row * N + (p * TILE_SIZE + tx)];
        } else {
            s_X[ty][tx] = 0.0; // Pad with 0s if we go off the edge
        }

        if (col < N && (p * TILE_SIZE + ty) < N) {
            s_W[ty][tx] = W[(p * TILE_SIZE + ty) * N + col];
        } else {
            s_W[ty][tx] = 0.0;
        }

        // --- SYNCHRONIZATION BARRIER ---
        // CRITICAL: We cannot do math until every thread has finished fetching its piece!
        __syncthreads();

        // --- THE MATH (Inner Loop) ---
        // Now all threads calculate the partial dot product using ONLY the ultra-fast SRAM.
        for (int k = 0; k < TILE_SIZE; ++k) {
            dot_product += s_X[ty][k] * s_W[k][tx];
        }

        // --- SYNCHRONIZATION BARRIER ---
        // CRITICAL: Wait for all threads to finish their math before the outer loop 
        // repeats and overwrites the SRAM with the next tile!
        __syncthreads();
    }

    // E. We have slid all the way across. Write the final answer to slow VRAM.
    if (row < N && col < N) {
        Y[row * N + col] = dot_product;
    }
}

// ---------------------------------------------------------
// 2. THE LAUNCHER (Runs on the CPU)
// ---------------------------------------------------------
void launch_tiled_matmul(const float* d_X, const float* d_W, float* d_Y, int N) {
    // Define the dimensions of our Thread Block (16x16 = 256 threads working together)
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    // Calculate how many 16x16 blocks we need to cover the entire matrix
    // Adding (TILE_SIZE - 1) ensures we round up if N isn't perfectly divisible by 16
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Trigger the execution on the GPU!
    tiled_matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_X, d_W, d_Y, N);
}