#include <iostream>
#include <vector>
#include <cassert>

// 1. Declare the FFI function we are testing
extern "C" void run_forward_pass(const float* host_X, const float* host_W, float* host_Y, int N);

int main() {
    int N = 2;
    std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> W = {2.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> Y(4, 0.0f); // Initialize output with 0s

    // 2. Call the function
    run_forward_pass(X.data(), W.data(), Y.data(), N);

    // 3. Assert the results (if assert fails, the program crashes immediately)
    assert(Y[0] == 4.0f);
    assert(Y[1] == 4.0f);
    assert(Y[2] == 10.0f);
    assert(Y[3] == 8.0f);

    std::cout << "\nSUCCESS: C++ CUDA Engine Math Test Passed!" << std::endl;
    return 0;
}