#include "book.h"
#include "cpu_bitmap.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"

constexpr int DIM = 5000;

struct cuComplex {
    float real;
    float image;

    __device__ cuComplex (float r, float i) : real (r), image (i) {
    }
    __device__ float magnitude2 () {
        return real * real + image * image;
    }
    __device__ cuComplex operator* (const cuComplex& c) {
        return cuComplex (real * c.real - image * c.image,
                          image * c.real + real * c.image);
    }
    __device__ cuComplex operator+ (const cuComplex& c) {
        return cuComplex (real + c.real, image + c.image);
    }
};

// __device__ macro will be only executed in device,
// called by __device__ or __global__ functions.
__device__ int julia (int x, int y) {
    constexpr float scale = 1.5;

    float jx = scale * static_cast<float> ((DIM >> 1) - x) / (DIM >> 1);
    float jy = scale * static_cast<float> ((DIM >> 1) - y) / (DIM >> 1);

    cuComplex c (-0.8, 0.156);
    cuComplex a (jx, jy);
    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2 () > 1000) {
            return 0;
        }
    }
    return 1;
}

// __global__ macro shares the function between CPU and GPU.
__global__ void kernel (unsigned char* ptr) {
    // Get the blockId for each pixel. For offset, we have ind = x + y * Dx + z * Dy * Dx.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * DIM;

    if (x >= DIM || y >= DIM)
        return;

    // Calculate julia value.
    int juliaVal = julia (x, y);

    // OpenCV with CV_8UC4 has the BGR\alpha image.
    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = 255 * juliaVal;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main () {
    // The device bitmap and the host bitmap, with space allocation.
    unsigned char* dev_bitmap = NULL;
    CPUBitmap bitmap (DIM, DIM, static_cast<void*> (dev_bitmap));
    HANDLE_ERROR (cudaMalloc (reinterpret_cast<void**> (&dev_bitmap), bitmap.image_size ()));

    // Using 2 dimension grid. Each block with 1 thread.
    constexpr int threadsSize = 32;
    dim3 gridDim (DIM / threadsSize + 1, DIM / threadsSize + 1);
    dim3 blockDim (threadsSize, threadsSize);
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