#include "book.h"
#include "cpu_bitmap.h"
#include "share_bitmap.cuh"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"

constexpr int DIM = 1024;
constexpr float PI = 3.1415926535897932f;


// __global__ macro shares the function between CPU and GPU.
__global__ void kernel (unsigned char* ptr) {
    // Get the blockId for each pixel. For offset, we have ind = x + y * Dx + z * Dy * Dx.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * DIM;

    if (x >= DIM || y >= DIM)
        return;

    __shared__ float shared[16][16];

    constexpr float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] = 255 * (sinf (x * 2.0f * PI / period) + 1.0f) *
    (sinf (y * 2.0f * PI / period) + 1.0f) / 4.0f;

    __syncthreads();

    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main () {
    // The device bitmap and the host bitmap, with space allocation.
    unsigned char* dev_bitmap = NULL;
    CPUBitmap bitmap (DIM, DIM, static_cast<void*> (dev_bitmap));
    HANDLE_ERROR (cudaMalloc (reinterpret_cast<void**> (&dev_bitmap), bitmap.image_size ()));

    // Using 2 dimension grid. Each block with 1 thread.
    dim3 gridDim (DIM / 16, DIM / 16);
    dim3 blockDim (16, 16);
    kernel<<<gridDim, blockDim>>> (dev_bitmap);

    // Copy the contents. Then free the space.
    HANDLE_ERROR (cudaMemcpy (bitmap.get_ptr (), dev_bitmap,
                              bitmap.image_size (), cudaMemcpyDeviceToHost));
    HANDLE_ERROR (cudaFree (dev_bitmap));

    // Save the image.
    cv::Mat image (DIM, DIM, CV_8UC4, bitmap.get_ptr ());
    cv::imwrite ("output.png", image);
    cv::imwrite ("output.jpg", image);
}