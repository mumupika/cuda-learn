#include "book.h"

#define N (33 * 1024)

__global__ void add (int* a, int* b, int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main () {
    int a[N], b[N], c[N];
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    int *dev_a, *dev_b, *dev_c;
    HANDLE_ERROR (cudaMalloc ((void**)&dev_a, sizeof (int) * N));
    HANDLE_ERROR (cudaMalloc ((void**)&dev_b, sizeof (int) * N));
    HANDLE_ERROR (cudaMalloc ((void**)&dev_c, sizeof (int) * N));

    HANDLE_ERROR (cudaMemcpy (dev_a, a, sizeof (int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR (cudaMemcpy (dev_b, b, sizeof (int) * N, cudaMemcpyHostToDevice));

    constexpr int UNIT = 128;
    add<<<UNIT, UNIT>>> (dev_a, dev_b, dev_c);

    HANDLE_ERROR (cudaMemcpy (c, dev_c, sizeof (int) * N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        printf ("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    return 0;
}