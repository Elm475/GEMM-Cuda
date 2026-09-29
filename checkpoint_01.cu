#include <stdio.h>

__global__ void matrixMul(float *a, float *b, float *c, int n){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float acc = 0.0f;
    for(int k = 0; k < n; k++){
        acc += a[row*n + k] * b[k*n + col];
    }
    c[row*n + col] = acc;
}

int main(){
    float *a_h, *b_h, *c_h, *a_d, *b_d, *c_d;
    a_h = (float*)malloc(8*8*sizeof(float));
    b_h = (float*)malloc(8*8*sizeof(float));
    c_h = (float*)malloc(8*8*sizeof(float));
    cudaMalloc((void**)&a_d, 8*8*sizeof(float));
    cudaMalloc((void**)&b_d, 8*8*sizeof(float));
    cudaMalloc((void**)&c_d, 8*8*sizeof(float));
    for(int i = 0; i < 8; i++)
        for(int j = 0; j < 8; j++){
            a_h[i*8+j] = 1.0f;
            b_h[i*8+j] = 2.0f;
        }
    cudaMemcpy(a_d, a_h, 8*8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, 8*8*sizeof(float), cudaMemcpyHostToDevice);
    matrixMul<<<1, dim3(8,8)>>>(a_d, b_d, c_d, 8);
    cudaMemcpy(c_h, c_d, 8*8*sizeof(float), cudaMemcpyDeviceToHost);
    printf("c[1][1] = %f\n", c_h[1*8+1]);
    printf("c[2][3] = %f\n", c_h[2*8+3]);
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
    free(a_h); free(b_h); free(c_h);
    return 0;
}
