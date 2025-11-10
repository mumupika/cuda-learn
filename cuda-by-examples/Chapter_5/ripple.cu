#include "book.h"
#include "cpu_anim.h"
#include "ripple.cuh"

void cleanup (DataBlock* d) {
    cudaFree (d->dev_bitmap);
}

void generate_frame (DataBlock* d, int ticks) {
    dim3 blocks (DIM / 16, DIM / 16);
    dim3 threads (16, 16);
    kernel<<<blocks, threads>>> (d->dev_bitmap, ticks);

    HANDLE_ERROR (cudaMemcpy (d->bitmap->get_ptr (), d->dev_bitmap,
                              d->bitmap->image_size (), cudaMemcpyDeviceToHost));
}

__global__ void kernel (unsigned char* ptr, int ticks) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x - static_cast<float> (DIM) / 2;
    float fy = y - static_cast<float> (DIM) / 2;
    float d = sqrtf (fx * fx + fy * fy);

    unsigned char grey =
    (unsigned char)(128.0f + 127.0f * cos (d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

    for (int i = 0; i < 3; i++) {
        ptr[offset * 4 + i] = grey;
    }
    ptr[offset * 4 + 3] = 255;
}

int main () {
    DataBlock data;
    CPUAnimBitmap bitmap (DIM, DIM, &data);
    data.bitmap = &bitmap;
    HANDLE_ERROR (cudaMalloc ((void**)(&data.dev_bitmap), bitmap.image_size ()));

    generate_frame(&data, 1);
    cleanup(&data);
}