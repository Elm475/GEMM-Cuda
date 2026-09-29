#include<stdio.h>
__global__ void write_id(int *a)
{
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;
    a[global_id] = global_id;
    printf("a[%d]\n",global_id);
}
int main()
{
    int a_h[10] = {0};
    int *a_d = 0;
    cudaMalloc((void**)&a_d, 10 * sizeof(int));
    cudaMemcpy(a_d, a_h, 10 * sizeof(int),cudaMemcpyHostToDevice);
    write_id<<<1,10>>>(a_d);
    cudaDeviceSynchronize(); 
    cudaMemcpy(a_h, a_d, 10 * sizeof(int),cudaMemcpyDeviceToHost);
    for(int i = 0; i<10; i++)
    printf("a_h[%d]=%d\n",i,a_h[i]);
    cudaFree(a_d);
}