#include "book.h"
#include "cpu_bitmap.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
constexpr float INF = 2e10f;
constexpr int SPHERES = 2285;
constexpr int DIM = 1024;
inline float rnd (float x) {
    return x * rand () / RAND_MAX;
}
struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;

    __device__ float hit (float ox, float oy, float* n) {
        float dx = ox - x;
        float dy = oy - y;

        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf (radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf (radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

Sphere* s;

__global__ void kernel (Sphere* s, unsigned char* ptr) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - static_cast<float>(DIM) / 2);
    float oy = (y - static_cast<float>(DIM) / 2);
    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit (ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }
    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}
int main () {
    // Capture the start time.
    cudaEvent_t start, stop;
    HANDLE_ERROR (cudaEventCreate (&start));
    HANDLE_ERROR (cudaEventCreate (&stop));
    HANDLE_ERROR (cudaEventRecord (start, 0));

    CPUBitmap bitmap (DIM, DIM);
    unsigned char* dev_bitmap;

    Sphere* temp_s = (Sphere*)malloc (sizeof (Sphere) * SPHERES);

    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd (1.0f);
        temp_s[i].g = rnd (1.0f);
        temp_s[i].b = rnd (1.0f);
        temp_s[i].x = rnd (1000.0f) - 500;
        temp_s[i].y = rnd (1000.0f) - 500;
        temp_s[i].z = rnd (1000.0f) - 500;
        temp_s[i].radius = rnd (100.0f) + 20;
    }
    HANDLE_ERROR (cudaMalloc ((void**)&dev_bitmap, bitmap.image_size ()));
    HANDLE_ERROR (cudaMalloc ((void**)&s, sizeof (Sphere) * SPHERES));
    HANDLE_ERROR (cudaMemcpy (s, temp_s, sizeof (Sphere) * SPHERES, cudaMemcpyHostToDevice));
    free (temp_s);

    // Generate bitmap from kernel.
    dim3 grids (DIM / 16, DIM / 16);
    dim3 threads (16, 16);
    kernel<<<grids, threads>>> (s, dev_bitmap);

    // Copy back. Dislay.
    // copy our bitmap back from the GPU for display
    HANDLE_ERROR (cudaMemcpy (bitmap.get_ptr (), dev_bitmap,
                              bitmap.image_size (), cudaMemcpyDeviceToHost));
    
    // get stop time, and display the timing results
    HANDLE_ERROR (cudaEventRecord (stop, 0));
    HANDLE_ERROR (cudaEventSynchronize (stop));
    float elapsedTime;
    HANDLE_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
    printf ("Time to generate: %3.1f ms\n", elapsedTime);
    HANDLE_ERROR (cudaEventDestroy (start));
    HANDLE_ERROR (cudaEventDestroy (stop));

    cv::Mat image (DIM, DIM, CV_8UC4, bitmap.get_ptr ());
    cv::imwrite ("output.png", image);
    cv::imwrite ("output.jpg", image);
    cudaFree (dev_bitmap);
    cudaFree (s);
}