#include <cstdio>

// A simple kernel that prints from each thread
__global__ void hello_kernel() {
    printf("Hello, World from GPU thread %d in block %d!\n",
           threadIdx.x, blockIdx.x);
}

int main() {
    // Launch 1 block of 8 threads
    hello_kernel<<<1, 8>>>();
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
    return 0;
}
