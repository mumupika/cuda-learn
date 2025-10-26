#ifndef RIPPLE_CUH_
#define RIPPLE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include "cpu_anim.h"

#define DIM 1024

struct DataBlock {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

void cleanup(DataBlock *d);
void generate_frame(DataBlock *d, int ticks);
__global__ void kernel(unsigned char *ptr, int ticks);

#endif