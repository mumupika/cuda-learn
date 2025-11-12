#include "book.h"
#include <cuda.h>
#include <stdio.h>


// Simple copy kernel
__global__ void copyKernel (float *d_data, float *d_data2, int width, int height) {
    // Calculate surface coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;
    if (x < width && y < height) {
        d_data2[offset] = sqrtf(d_data[offset]) + 1.0f;
    }
}

// Host code
int main () {
    const int height = 6700;
    const int width = 6700;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);

    float* h_data = (float*)malloc (sizeof (float) * width * height);
    for (int i = 0; i < height * width; ++i) {
        h_data[i] = i;
    }

    float *d_data, *d_data2;
    HANDLE_ERROR(cudaMalloc((void **)&d_data, sizeof(float) * height * width));
    HANDLE_ERROR(cudaMalloc((void **)&d_data2, sizeof(float) * height * width));
    HANDLE_ERROR(cudaMemcpy(d_data, h_data, sizeof(float) * height * width, cudaMemcpyHostToDevice));

    // Invoke kernel
    dim3 threadsperBlock (32, 32);
    dim3 numBlocks ((width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (height + threadsperBlock.y - 1) / threadsperBlock.y);

    cudaEvent_t start, stop;
    HANDLE_ERROR (cudaEventCreate (&start));
    HANDLE_ERROR (cudaEventCreate (&stop));

    HANDLE_ERROR (cudaEventRecord (start));
    copyKernel<<<numBlocks, threadsperBlock>>> (d_data, d_data2, width, height);
    HANDLE_ERROR (cudaEventRecord (stop));
    HANDLE_ERROR (cudaEventSynchronize (stop));

    float time;
    HANDLE_ERROR (cudaEventElapsedTime (&time, start, stop));

    printf ("Elapsed time: %0.3f ms\n", time);

    return 0;
}
