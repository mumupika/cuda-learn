#include "book.h"
#include "cpu_anim.h"
#include "cuda.h"
#include <cstdio>

constexpr int DIM = 256;
constexpr float MAX_TEMP = 1.0f;
constexpr float MIN_TEMP = 0.0001f;
constexpr float SPEED = 0.25f;

cudaTextureObject_t texIn, texConst;
cudaArray_t cuInArray, cuConstArray;

// globals needed by update routine.
struct DataBlock {
    unsigned char* output_bitmap;
    float* dev_inSrc;
    float* dev_outSrc;
    float* dev_constSrc;
    CPUAnimBitmap* bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};
// cptr can be texturized.
__global__ void copy_const_kernel (float* iptr, cudaTextureObject_t cptr) {
    // map by threadIdx / blockIdx to pixel.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (tex1D<float> (cptr, offset) != 0) {
        iptr[offset] = tex1D<float> (cptr, offset);
    }
}
// inSrc can be texturized.
__global__ void blend_kernel (float* outSrc, cudaTextureObject_t inSrc) {
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

    outSrc[offset] = tex1D<float> (inSrc, offset) +
    SPEED *
    (tex1D<float> (inSrc, top) + tex1D<float> (inSrc, bottom) + tex1D<float> (inSrc, left) +
     tex1D<float> (inSrc, right) - 4 * tex1D<float> (inSrc, offset));
}
void anim_gpu (DataBlock* d, int ticks) {
    HANDLE_ERROR (cudaEventRecord (d->start, 0));
    dim3 blocks (DIM / 16, DIM / 16);
    dim3 threads (16, 16);
    CPUAnimBitmap* bitmap = d->bitmap;

    for (int i = 0; i < 90; i++) {
        cudaMemcpy2DToArray (cuInArray, 0, 0, d->dev_inSrc, bitmap->image_size (),
                             bitmap->image_size (), 1, cudaMemcpyDeviceToDevice);
        copy_const_kernel<<<blocks, threads>>> (d->dev_inSrc, texConst);
        cudaMemcpy2DToArray (cuInArray, 0, 0, d->dev_inSrc, bitmap->image_size (),
                             bitmap->image_size (), 1, cudaMemcpyDeviceToDevice);
        blend_kernel<<<blocks, threads>>> (d->dev_outSrc, texIn);
        swap (d->dev_inSrc, d->dev_outSrc);
    }
    float_to_color<<<blocks, threads>>> (d->output_bitmap, d->dev_inSrc);
    HANDLE_ERROR (cudaMemcpy (bitmap->get_ptr (), d->output_bitmap,
                              bitmap->image_size (), cudaMemcpyDeviceToHost));
    HANDLE_ERROR (cudaEventRecord (d->stop, 0));
    cudaDeviceSynchronize();
    float elapsedTime;
    HANDLE_ERROR (cudaEventElapsedTime (&elapsedTime, d->start, d->stop));
    d->totalTime += elapsedTime;
    d->frames++;
    printf ("Average Time per frame: %3.1f ms\n", d->totalTime / d->frames);
}
void anim_exit (DataBlock* d) {
    cudaFree (d->dev_inSrc);
    cudaFree (d->dev_outSrc);
    cudaFree (d->dev_constSrc);

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
    HANDLE_ERROR (cudaMalloc ((void**)&data.dev_inSrc, bitmap.image_size ()));
    HANDLE_ERROR (cudaMalloc ((void**)&data.dev_outSrc, bitmap.image_size ()));
    HANDLE_ERROR (cudaMalloc ((void**)&data.dev_constSrc, bitmap.image_size ()));

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
    cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc (32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray (&cuConstArray, &channelDesc, DIM * DIM);
    cudaMemcpy2DToArray (cuConstArray, 0, 0, temp, bitmap.image_size (),
                         bitmap.image_size (), 0, cudaMemcpyHostToDevice);
    struct cudaResourceDesc resConstDesc;
    memset (&resConstDesc, 0, sizeof (resConstDesc));
    resConstDesc.res.array.array = cuConstArray;
    resConstDesc.resType = cudaResourceTypeArray;
    cudaCreateTextureObject (&texConst, &resConstDesc, NULL, NULL);

    // initialize the input data
    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }

    // HANDLE_ERROR (cudaMemcpy (data.dev_inSrc, temp, bitmap.image_size (), cudaMemcpyHostToDevice));
    cudaChannelFormatDesc channelDesc2 =
    cudaCreateChannelDesc (32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray (&cuInArray, &channelDesc2, DIM * DIM);
    cudaMemcpy2DToArray (cuInArray, 0, 0, temp, bitmap.image_size (),
                         bitmap.image_size (), 0, cudaMemcpyHostToDevice);
    struct cudaResourceDesc resInDesc;
    memset (&resInDesc, 0, sizeof (resInDesc));
    resInDesc.res.array.array = cuInArray;
    resInDesc.resType = cudaResourceTypeArray;
    cudaCreateTextureObject (&texIn, &resInDesc, NULL, NULL);

    free (temp);
    anim_gpu (&data, 1);
    anim_exit (&data);
    return 0;
}