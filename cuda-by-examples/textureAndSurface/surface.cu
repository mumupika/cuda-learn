#include "book.h"
#include <cmath>
#include <cuda.h>
#include <stdio.h>

// Simple copy kernel
__global__ void copyKernel (cudaSurfaceObject_t inputSurfObj,
                            cudaSurfaceObject_t outputSurfObj,
                            int width,
                            int height) {
    // Calculate surface coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float data;
        // Read from input surface
        data = surf2Dread<float> (inputSurfObj, x * sizeof (float), y);
        // Write to output surface
        data = sqrtf (data);
        surf2Dwrite<float> (data, outputSurfObj, x * sizeof (float), y);
    }
}

// Host code
int main () {
    const int height = 8192;
    const int width = 8192;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);

    // Allocate and set some host data
    float* h_data = (float*)malloc (sizeof (float) * width * height);
    for (int i = 0; i < height * width; ++i) {
        h_data[i] = i;
    }

    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc (32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t cuInputArray;
    cudaMallocArray (&cuInputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);
    cudaArray_t cuOutputArray;
    cudaMallocArray (&cuOutputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);

    // Set pitch of the source (the width in memory in bytes of the 2D array
    // pointed to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof (float);
    // Copy data located at address h_data in host memory to device memory
    cudaMemcpy2DToArray (cuInputArray, 0, 0, h_data, spitch,
                         width * sizeof (float), height, cudaMemcpyHostToDevice);

    // Specify surface
    struct cudaResourceDesc resDesc;
    memset (&resDesc, 0, sizeof (resDesc));
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface objects
    resDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject (&inputSurfObj, &resDesc);
    resDesc.res.array.array = cuOutputArray;
    cudaSurfaceObject_t outputSurfObj = 0;
    cudaCreateSurfaceObject (&outputSurfObj, &resDesc);

    // Invoke kernel
    dim3 threadsperBlock (32, 32);
    dim3 numBlocks ((width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (height + threadsperBlock.y - 1) / threadsperBlock.y);

    copyKernel<<<numBlocks, threadsperBlock>>> (inputSurfObj, outputSurfObj, width, height);

    cudaEvent_t start, stop;
    HANDLE_ERROR (cudaEventCreate (&start));
    HANDLE_ERROR (cudaEventCreate (&stop));

    HANDLE_ERROR (cudaEventRecord (start));
    for (int i = 0; i < 99; i++) {
        copyKernel<<<numBlocks, threadsperBlock>>> (inputSurfObj, outputSurfObj,
                                                    width, height);
    }
    HANDLE_ERROR (cudaEventRecord (stop));
    HANDLE_ERROR (cudaEventSynchronize (stop));

    float time;
    HANDLE_ERROR (cudaEventElapsedTime (&time, start, stop));

    printf ("Elapsed time: %0.3f ms\n", time);

    // Copy data from device back to host
    cudaMemcpy2DFromArray (h_data, spitch, cuOutputArray, 0, 0,
                           width * sizeof (float), height, cudaMemcpyDeviceToHost);

    // Destroy surface objects
    cudaDestroySurfaceObject (inputSurfObj);
    cudaDestroySurfaceObject (outputSurfObj);

    // Free device memory
    cudaFreeArray (cuInputArray);
    cudaFreeArray (cuOutputArray);

    // Free host memory
    free (h_data);

    return 0;
}
