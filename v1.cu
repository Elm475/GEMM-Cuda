#include <stdio.h>

__global__ void matrixMul(float *a, float *b, float *c, int n){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    float acc = 0.0f;
    for(int k = 0; k < n; k++)
        acc += a[row*n + k] * b[k*n + col];
    c[row*n + col] = acc;
}

int main(int argc, char *argv[]){
    int N = atoi(argv[1]);
    float *a_h, *b_h, *c_h, *a_d, *b_d, *c_d;
    a_h = (float*)malloc(N*N*sizeof(float));
    b_h = (float*)malloc(N*N*sizeof(float));
    c_h = (float*)malloc(N*N*sizeof(float));
    cudaMalloc((void**)&a_d, N*N*sizeof(float));
    cudaMalloc((void**)&b_d, N*N*sizeof(float));
    cudaMalloc((void**)&c_d, N*N*sizeof(float));
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++){
            a_h[i*N+j] = 1.0f;
            b_h[i*N+j] = 2.0f;
        }
    }
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    cudaMemcpy(a_d, a_h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    matrixMul<<<grid, block>>>(a_d, b_d, c_d, N);
    cudaMemcpy(c_h, c_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("c[1][1] = %f\n", c_h[1*N+1]);
    printf("c[2][3] = %f\n", c_h[2*N+3]);
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
    free(a_h); free(b_h); free(c_h);
    return 0;
}
