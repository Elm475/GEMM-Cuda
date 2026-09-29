#include<stdio.h>
__global__ void kernel(){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block_id:%d, thread_id:%d, globalid:%d\n", blockIdx.x, threadIdx.x, global_id);
}
int main(){
    kernel<<<2,8>>>();
    cudaDeviceSynchronize();
    return 0;
}