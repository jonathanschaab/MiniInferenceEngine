// 1. We define the exact C++ function signature we want to call.
// The "C" tells Rust to use the standard C-ABI (Application Binary Interface).
unsafe extern "C" {
    fn run_forward_pass(
        host_X: *const f32, 
        host_W: *const f32, 
        host_Y: *mut f32, 
        N: i32
    );
}

// 2. We create a safe Rust wrapper around the unsafe C++ call
pub fn execute_gpu_math(batch_matrix: Vec<f32>, weight_matrix: Vec<f32>, matrix_size: i32) -> Vec<f32> {
    // A. Create an empty Vector of zeroes to hold the output answers from the GPU
    let total_elements = (matrix_size * matrix_size) as usize;
    let mut output_matrix = vec![0.0; total_elements];

    // B. The Unsafe Boundary!
    // We are trusting the C++ DLL not to crash or access bad memory.
    unsafe {
        run_forward_pass(
            batch_matrix.as_ptr(),       // *const f32 (Read-only pointer)
            weight_matrix.as_ptr(),      // *const f32 (Read-only pointer)
            output_matrix.as_mut_ptr(),  // *mut f32   (Mutable pointer for the answer)
            matrix_size
        );
    }

    // C. Return the populated answers back to safe Rust!
    output_matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_math_bridge() {
        // 1. Define our 2x2 test matrices
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![2.0, 0.0, 1.0, 2.0];
        let matrix_size = 2;

        // 2. Call our safe Rust wrapper (which triggers the C++ DLL over FFI)
        let y = execute_gpu_math(x, w, matrix_size);

        // 3. Assert the results match our hand-calculated math exactly
        assert_eq!(y, vec![4.0, 4.0, 10.0, 8.0]);
    }
}