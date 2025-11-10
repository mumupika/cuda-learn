#include "book.h"
#include <cuda.h>
#include <stdio.h>


// Simple copy kernel
__global__ void copyKernel (float* d_data2, cudaTextureObject_t d_data, int width, int height) {
    // Calculate surface coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;
    if (x < width && y < height) {
        d_data2[offset] = sqrtf (tex2D<float> (d_data, x, y)) + 1.0f;
    }
}

// Host code
int main () {
    const int height = 1024;
    const int width = 1024;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);

    float* h_data = (float*)malloc (sizeof (float) * width * height);
    for (int i = 0; i < height * width; ++i) {
        h_data[i] = i;
    }

    float* d_data2;
    HANDLE_ERROR (cudaMalloc ((void**)&d_data2, sizeof (float) * height * width));


    cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc (32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray (&cuArray, &channelDesc, width, height);
    const size_t spitch = width * sizeof (float);
    // Copy data located at address h_data in host memory to device memory
    cudaMemcpy2DToArray (cuArray, 0, 0, h_data, spitch, width * sizeof (float),
                         height, cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset (&resDesc, 0, sizeof (resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureObject_t d_data;
    cudaCreateTextureObject(&d_data, &resDesc, NULL, NULL);

    // Invoke kernel
    dim3 threadsperBlock (32, 32);
    dim3 numBlocks ((width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (height + threadsperBlock.y - 1) / threadsperBlock.y);


    copyKernel<<<numBlocks, threadsperBlock>>> (d_data2, d_data, width, height);


    cudaEvent_t start, stop;
    HANDLE_ERROR (cudaEventCreate (&start));
    HANDLE_ERROR (cudaEventCreate (&stop));

    HANDLE_ERROR (cudaEventRecord (start));
    for (int i = 0; i < 99; i++) {
        copyKernel<<<numBlocks, threadsperBlock>>> (d_data2, d_data, width, height);
    }
    HANDLE_ERROR (cudaEventRecord (stop));
    HANDLE_ERROR (cudaEventSynchronize (stop));

    float time;
    HANDLE_ERROR (cudaEventElapsedTime (&time, start, stop));

    printf ("Elapsed time: %0.3f ms\n", time);

    return 0;
}
