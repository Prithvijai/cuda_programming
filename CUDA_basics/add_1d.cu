#include <iostream>
#include <cuda_runtime.h>

// #define N 1024  // Vector size = 10 million (number of elements in both vectors)
#define BLOCK_SIZE_1D 256    // since i am using 1D grid i will use 1D block

__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;     // assinging thread index
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char *argv[]) {

    int N = 1024; // Default size
    std::cout<<"CUDA vector adder. Enter the Size of the array A, and B"<<std::endl;

    std::cin>>N;

    float *h_A, *h_B, *h_c_gpu; // host vectors
    float *d_A, *d_B, *d_C; // device vectors pointers
    size_t size = N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    std::cout<<"Vector size: " << N << std::endl;
    // std::cout<<"Point d_A: " << &d_A << std::endl;
    // std::cout<<"address of d_A: " << d_A << std::endl;
    

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // std::cout<<"point **d_A: " << &d_A << std::endl;

    init_vector(h_A, N);
    init_vector(h_B, N);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    std::cout<<"Number of blocks: " << num_blocks << std::endl;

    // called kernel to perform vector addition
    vector_add_gpu_1d<<<num_blocks, BLOCK_SIZE_1D>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();  // wait for all the threads to finish

    cudaMemcpy(h_c_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // check for errors using cpu
    for (int i = 0; i < N; i++) {
        if (h_c_gpu[i] != h_A[i] + h_B[i]) {
            std::cout << "Error at index " << i << ": " << h_c_gpu[i] << " != " << h_A[i] + h_B[i] << std::endl;
            break;
        }
        else {
            std::cout<< h_c_gpu[i] << std::endl;
        }
    }

    free(h_A);
    free(h_B);
    free(h_c_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    std::cout<<"Freeing memory..."<<std::endl;
    return 0;
}