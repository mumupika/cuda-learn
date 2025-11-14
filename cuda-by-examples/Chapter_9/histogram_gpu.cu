#include "book.h"
#include <cuda_runtime.h>
constexpr int SIZE = 100 * 1024 * 1024;

__global__ void histo_kernel (unsigned char* buffer, int size, unsigned int* dev_histogram) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while(i < size) {
        atomicAdd(&dev_histogram[buffer[i]], 1);
        i += stride;
    }
}

int main () {
    unsigned char* buffer = (unsigned char*)big_random_block (SIZE);
    cudaEvent_t start, stop;
    HANDLE_ERROR (cudaEventCreate (&start));
    HANDLE_ERROR (cudaEventCreate (&stop));
    HANDLE_ERROR (cudaEventRecord (start, 0));

    unsigned char* dev_buffer;
    unsigned int* dev_histogram;

    HANDLE_ERROR (cudaMalloc ((void**)&dev_buffer, SIZE));
    HANDLE_ERROR (cudaMemcpy (dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));

    HANDLE_ERROR (cudaMalloc ((void**)&dev_histogram, 256 * sizeof (unsigned int)));
    HANDLE_ERROR (cudaMemset (dev_histogram, 0, 256 * sizeof (int)));

    cudaDeviceProp prop;
    HANDLE_ERROR (cudaGetDeviceProperties (&prop, 0));

    int blocks = prop.multiProcessorCount;
    histo_kernel<<<blocks * 2, 256, 0, 0>>> (dev_buffer, SIZE, dev_histogram);
    unsigned int histogram[256];
    HANDLE_ERROR (cudaMemcpy (histogram, dev_histogram, 256 * sizeof (int),
                              cudaMemcpyDeviceToHost));

    HANDLE_ERROR (cudaEventRecord (stop, 0));
    HANDLE_ERROR (cudaEventSynchronize (stop));
    float ElapsedTime;
    HANDLE_ERROR (cudaEventElapsedTime (&ElapsedTime, start, stop));
    printf ("Time to generate: %3.1f ms\n", ElapsedTime);

    long long histoCnt = 0;
    for (int i = 0; i < 256; i++) {
        histoCnt += histogram[i];
    }
    printf ("Histogram Sum: %lld\n", histoCnt);

    // verify that we have the same counts via CPU
    for (int i = 0; i < SIZE; i++) {
        histogram[buffer[i]]--;
    }
    for (int i = 0; i < 256; i++) {
        if (histogram[i] != 0) {
            printf ("Failure at %d!\n", i);
        }
    }
    HANDLE_ERROR (cudaEventDestroy (start));
    HANDLE_ERROR (cudaEventDestroy (stop));
    return 0;
}