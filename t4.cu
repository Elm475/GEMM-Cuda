#include<stdio.h>
#define N 1000
#define BLOCK_SIZE 256
__global__ void martrixAdd(float *a, float *b, float *c, int n)
{
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < N && col < N)
    c[row * N + col] = a[row * N + col] + b[row * N + col];
}
int main()
{
    float *a_h, *b_h, *c_h, *c_d, *a_d, *b_d;
    a_h = (float *)(malloc(N * N * sizeof(float)));
    b_h = (float *)(malloc(N * N * sizeof(float)));
    c_h = (float *)(malloc(N * N * sizeof(float)));
    cudaMalloc((void **)&a_d,N * N * sizeof(float));
    cudaMalloc((void **)&b_d,N * N * sizeof(float));
    cudaMalloc((void **)&c_d,N * N * sizeof(float));
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            a_h[i*N + j] = i+j;
            b_h[i*N + j] = (i+j);
        }
    }
    cudaMemcpy(a_d, a_h, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, N * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid(N-16+1/16, N-16+1/16);
    martrixAdd<<<grid,block>>>(a_d, b_d, c_d, N);
    cudaMemcpy(c_h, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("c_h[1][1] = %f\n", c_h[1*N + 1]);
    printf("c_h[999][998] = %f\n", c_h[999*N + 998]);
    cudaFree(a_d), cudaFree(b_d), cudaFree(c_d);
    free(a_h), free(b_h), free(c_h);
    return 0;
}