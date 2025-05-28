#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// 3*2 2*4  = 3*4  (m =3, k = 2 , n=4 )
#define BLOCK_SIZE 32

// a, b, c are allocated has 1D but there are truely 2D

void matmul_cpu(float *a, float *b, float *c, int m, int k, int n) {
    for(int i =0; i<m; i++)
    {
        for(int j = 0; j<n; j++)
        {
            float sum = 0.0;
            for (int x=0; x<k; x++)
            {
                sum += a[i * k + x] * b[x * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// covert that function into gpu thread running 
__global__ void matmul_gpu(float *a, float *b, float *c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (row <m && col < n){
        float sum = 0.0;
        for (int i=0; i < k; i++)
        {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}


int main() {

        int m, k, n;
        std::cout<<"Enter the m * k x k * n (m, k, n):"<<std::endl;

        std::cin>>m>>k>>n;

        float *a, *b, *c;
        a = (float*)malloc(m * k * sizeof(float));
        b = (float*)malloc(k * n * sizeof(float));
        c = (float*)malloc(m * n * sizeof(float));

        std::cout<<"Enter the A:"<<std::endl;
        for(int i=0; i< m ; i++)
        {
            for (int j=0; j< k ; j++)
            {
                std::cin>>a[i * k + j];
            }
        }
        std::cout<<"Enter the B:"<<std::endl;
        for(int i=0; i< k ; i++)
        {
            for (int j=0; j< n ; j++)
            {
                std::cin>>b[i * n + j];
            }
        }
        std::cout<<"The Matrix C is :"<<std::endl;

        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, m * k * sizeof(float));
        cudaMalloc(&d_B, k * n * sizeof(float));
        cudaMalloc(&d_C, m * n * sizeof(float));

        cudaMemcpy(d_A, a,m * k * sizeof(float) ,cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, b,k * n * sizeof(float) ,cudaMemcpyHostToDevice);

        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_C,m * n * sizeof(float) ,cudaMemcpyDeviceToHost);

        for(int i=0; i< m ; i++)
        {
            for (int j=0; j< n ; j++)
            {
                std::cout<<c[i * n + j]<< " ";
            }
            std::cout<<std::endl;
        }

        free(a);
        free(b);
        free(c);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return 0;


}
