#include <cuda_runtime.h>
#include <stdio.h>

/**
  * The float *host has: [0,1,2,...,63].
  * The cuArray will managed to make:
    [
        [0, 1, 2, ..., 7],
        [8, 9, 10, ..., 15],
        ...
        [56, 57, 58, ..., 63]
    ]
  * The width of the copy should be the padded width of the cuArray. We have 0 -> 7 is width, 0 -> 56 is height.
    The width is always in bytes, and the height is in the unit of row number.
  * The simple copy 2D to Texture Array then back. Result should be: [18, 19, 20, 26, 27, 28, 34, 35, 36].
 */
int main () {
    constexpr int W = 8, H = 8;
    float* host = (float*)malloc (sizeof (float) * W * H);
    for (int i = 0; i < W * H; i++) {
        host[i] = float (i);
        printf ("%1.f ", host[i]);
    }
    printf ("\n\n\n");
    cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc (32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray (&cuArray, &channelDesc, W, H);
    constexpr size_t spitch = W * sizeof (float);
    cudaMemcpy2DToArray (cuArray, 0, 0, host, spitch, W * sizeof (float), H,
                         cudaMemcpyHostToDevice);

    float* host_res = (float*)malloc (sizeof (float) * 3 * 3);
    cudaMemcpy2DFromArray (host_res, sizeof (float) * 3, cuArray, sizeof (float) * 2,
                           2, sizeof (float) * 3, 3, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 9; i++) {
        printf ("%1.f ", host_res[i]);
    }
    printf ("\n\n\n");
    float* host_res2 = (float*)malloc (sizeof (float) * H * W);
    cudaMemcpy2DFromArray (host_res2, W * sizeof (float), cuArray, 0, 0,
                           W * sizeof (float), H, cudaMemcpyDeviceToHost);
    for (int i = 0; i < H * W; i++) {
        printf ("%1.f ", host_res2[i]);
    }
    printf ("\n\n\n");
    float* host_res3 = (float*)malloc (sizeof (float) * 3 * 5);
    cudaMemcpy2DFromArray (host_res3, 5 * sizeof (float), cuArray, 2 * sizeof (float),
                           1, 5 * sizeof (float), 3, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3 * 5; i++) {
        printf ("%1.f ", host_res3[i]);
    }
    printf ("\n");
    return 0;
}
