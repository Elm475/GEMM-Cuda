#include<stdio.h>
#define N 400000
#define BLOCK_SIZE 256
__global__ void vectoradd(int *a, int *b, int *c, int n){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n)   
    c[global_id] = a[global_id] + b[global_id];
}
int main(){
    int *a_h, *b_h, *c_h, *a_d, *b_d, *c_d;
    a_h = (int *)malloc(N * sizeof(int));
    b_h = (int *)malloc(N * sizeof(int));
    c_h = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        a_h[i] = i;
        b_h[i] = i * 2;
    }
    cudaMalloc((void **)&a_d, N * sizeof(int));
    cudaMalloc((void **)&b_d, N * sizeof(int));
    cudaMalloc((void **)&c_d, N * sizeof(int));
    cudaMemcpy(a_d, a_h, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, N * sizeof(int), cudaMemcpyHostToDevice);
    int grid = (N + (BLOCK_SIZE) -1)/BLOCK_SIZE;
    vectoradd<<<grid, BLOCK_SIZE>>>(a_d, b_d, c_d, N);
    cudaMemcpy(c_h, c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("c[0]=%d,c[N-1] = %d",c_h[0],c_h[N-1]);
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
    free(a_h); free(b_h); free(c_h);
    return 0;
}