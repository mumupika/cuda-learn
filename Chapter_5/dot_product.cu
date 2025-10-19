#include "book.h"

#define imin(a, b) (a < b ? a : b)
#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)

constexpr int N = 33 * 1024;
constexpr int threadsPerBlock = 1024;

__global__ void dot (float* a, float* b, float* c) {
    // The buffer of shared memory.
    __shared__ float cache[threadsPerBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp = a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads ();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads ();
        i /= 2;
    }
    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

constexpr int blocksPerGrid = imin (32, (N + threadsPerBlock - 1) / threadsPerBlock);

int main () {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = new float[N];
    b = new float[N];
    partial_c = new float[N];

    HANDLE_ERROR (cudaMalloc ((void**)&dev_a, sizeof (float) * N));
    HANDLE_ERROR (cudaMalloc ((void**)&dev_b, sizeof (float) * N));
    HANDLE_ERROR (cudaMalloc ((void**)&dev_partial_c, sizeof (float) * blocksPerGrid));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR (cudaMemcpy (dev_a, a, N * sizeof (float), cudaMemcpyHostToDevice));
    HANDLE_ERROR (cudaMemcpy (dev_b, b, N * sizeof (float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>> (dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR (cudaMemcpy (partial_c, dev_partial_c, blocksPerGrid * sizeof (float),
                              cudaMemcpyDeviceToHost));
    
    // finish up on the CPU side  
    c = 0;  
    for (int i=0; i<blocksPerGrid; i++) {  
        c += partial_c[i];  
    }

    printf ("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares ((float)(N - 1))); // free memory on the GPU side
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_partial_c); // free memory on the CPU side
    delete[] a;
    delete[] b;
    delete[] partial_c;
    return 0;
}