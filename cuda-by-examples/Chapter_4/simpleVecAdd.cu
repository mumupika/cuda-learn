#include "book.h"

#define N 8200

__global__ void add (int* a, int* b, int* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
    return;
}

int main () {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate on the GPU.
    HANDLE_ERROR (cudaMalloc ((void**)&dev_a, N * sizeof (int)));
    HANDLE_ERROR (cudaMalloc ((void**)&dev_b, N * sizeof (int)));
    HANDLE_ERROR (cudaMalloc ((void**)&dev_c, N * sizeof (int)));

    // Fill array 'a' and 'b' on the CPU.
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * 2;
    }

    // copy data from host to device.
    HANDLE_ERROR (cudaMemcpy (dev_a, a, N * sizeof (int), cudaMemcpyHostToDevice));
    HANDLE_ERROR (cudaMemcpy (dev_b, b, N * sizeof (int), cudaMemcpyHostToDevice));

    // Kernel.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int gridSize = N / prop.maxThreadsPerBlock + 1;         // round up.
    add<<<gridSize, prop.maxThreadsPerBlock>>> (dev_a, dev_b, dev_c);

    // Copy the array back to host.
    HANDLE_ERROR (cudaMemcpy (c, dev_c, N * sizeof (int), cudaMemcpyDeviceToHost));

    // Display the result.
    for (int i = 0; i < N; i++) {
        printf ("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // free all device mem.
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);

    return 0;
}