#include "book.h"
#include "cpu_anim.h"
#include "cuda.h"
#include <cstdio>

constexpr int DIM = 256;
constexpr float MAX_TEMP = 1.0f;
constexpr float MIN_TEMP = 0.0001f;
constexpr float SPEED = 0.25f;

cudaArray_t cudaInArray, cudaOutArray, cudaConstArray;

// globals needed by update routine.
struct DataBlock {
    unsigned char* output_bitmap;
    cudaSurfaceObject_t surfIn, surfOut, surfConst;
    CPUAnimBitmap* bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};
__global__ void copy_const_kernel (cudaSurfaceObject_t iptr, cudaSurfaceObject_t cptr) {
    // map by threadIdx / blockIdx to pixel.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float num = surf1Dread<float> (cptr, offset);
    if (num != 0) {
        surf1Dwrite (num, iptr, offset);
    }
}
__global__ void float2color (unsigned char* optr, const cudaSurfaceObject_t outSrc) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = surf1Dread<float> (outSrc,offset);
    float s = 1;
    int h = (180 + (int)(360.0f * surf1Dread<float> (outSrc,offset))) % 360;
    float m1, m2;

    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    optr[offset * 4 + 0] = value (m1, m2, h + 120);
    optr[offset * 4 + 1] = value (m1, m2, h);
    optr[offset * 4 + 2] = value (m1, m2, h - 120);
    optr[offset * 4 + 3] = 255;
}
__global__ void blend_kernel (cudaSurfaceObject_t outSrc, cudaSurfaceObject_t inSrc) {
    // map by threadIdx / blockIdx to pixel.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;

    if (x == 0) {
        left++;
    }
    if (x == DIM - 1) {
        right--;
    }

    int top = offset - DIM, bottom = offset + DIM;
    if (y == 0) {
        top += DIM;
    }
    if (y == DIM - 1) {
        bottom -= DIM;
    }

    float input = surf1Dread<float> (inSrc, offset) +
    SPEED *
    (surf1Dread<float> (inSrc, top) + surf1Dread<float> (inSrc, bottom) +
     surf1Dread<float> (inSrc, left) + surf1Dread<float> (inSrc, right) -
     4 * surf1Dread<float> (inSrc, offset));
    surf1Dwrite (input, outSrc, offset);
}
void anim_gpu (DataBlock* d, int ticks) {
    HANDLE_ERROR (cudaEventRecord (d->start, 0));
    dim3 blocks (DIM / 16, DIM / 16);
    dim3 threads (16, 16);
    CPUAnimBitmap* bitmap = d->bitmap;

    for (int i = 0; i < 1000; i++) {
        copy_const_kernel<<<blocks, threads>>> (d->surfIn, d->surfConst);
        blend_kernel<<<blocks, threads>>> (d->surfOut, d->surfIn);

        cudaSurfaceObject_t temp;
        temp = d->surfIn;
        d->surfIn = d->surfOut;
        d->surfOut = temp;
    }
    
    float2color<<<blocks, threads>>> (d->output_bitmap, d -> surfOut);
    HANDLE_ERROR (cudaMemcpy (bitmap->get_ptr (), d->output_bitmap,
                              bitmap->image_size (), cudaMemcpyDeviceToHost));
    HANDLE_ERROR (cudaEventRecord (d->stop, 0));

    cudaDeviceSynchronize ();
    float elapsedTime;
    HANDLE_ERROR (cudaEventElapsedTime (&elapsedTime, d->start, d->stop));
    d->totalTime += elapsedTime;
    d->frames++;
    printf ("Average Time per frame: %3.1f ms\n", d->totalTime / d->frames);
}
void anim_exit (DataBlock* d) {
    cudaDestroySurfaceObject (d->surfConst);
    cudaDestroySurfaceObject (d->surfIn);
    cudaDestroySurfaceObject (d->surfOut);

    cudaFreeArray (cudaConstArray);
    cudaFreeArray (cudaInArray);
    cudaFreeArray (cudaOutArray);

    cudaFree (d->output_bitmap);

    HANDLE_ERROR (cudaEventDestroy (d->start));
    HANDLE_ERROR (cudaEventDestroy (d->stop));
}
int main () {
    DataBlock data;
    CPUAnimBitmap bitmap (DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR (cudaEventCreate (&data.start));
    HANDLE_ERROR (cudaEventCreate (&data.stop));

    HANDLE_ERROR (cudaMalloc ((void**)&data.output_bitmap, bitmap.image_size ()));

    cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc (32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray (&cudaInArray, &channelDesc, DIM * DIM, 0);
    cudaMallocArray (&cudaOutArray, &channelDesc, DIM * DIM, 0);
    cudaMallocArray (&cudaConstArray, &channelDesc, DIM * DIM, 0);

    struct cudaResourceDesc resInDesc, resConstDesc, resOutDesc;
    memset (&resInDesc, 0, sizeof (resInDesc));
    memset (&resOutDesc, 0, sizeof (resOutDesc));
    memset (&resConstDesc, 0, sizeof (resConstDesc));

    resInDesc.resType = cudaResourceTypeArray;
    resInDesc.res.array.array = cudaInArray;
    resOutDesc.resType = cudaResourceTypeArray;
    resOutDesc.res.array.array = cudaOutArray;
    resConstDesc.resType = cudaResourceTypeArray;
    resConstDesc.res.array.array = cudaConstArray;

    float* temp = (float*)malloc (bitmap.image_size ());
    for (int i = 0; i < DIM * DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601)) {
            temp[i] = MAX_TEMP;
        }
    }

    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;

    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }

    // HANDLE_ERROR (cudaMemcpy (data.dev_constSrc, temp, bitmap.image_size (), cudaMemcpyHostToDevice));
    cudaMemcpy2DToArray (cudaConstArray, 0, 0, temp, bitmap.image_size (),
                         bitmap.image_size (), 0, cudaMemcpyHostToDevice);

    // initialize the input data
    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }

    // HANDLE_ERROR (cudaMemcpy (data.dev_inSrc, temp, bitmap.image_size (), cudaMemcpyHostToDevice));
    cudaMemcpy2DToArray (cudaInArray, 0, 0, temp, bitmap.image_size (),
                         bitmap.image_size (), 0, cudaMemcpyHostToDevice);

    free (temp);
    anim_gpu (&data, 1);
    anim_exit (&data);
    return 0;
}