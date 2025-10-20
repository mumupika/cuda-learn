# Cuda by example Notes

## 3 Intro to CUDA C

### Target

[] Write CUDA C for the first time.
[] Difference between code for **host** and **device**.
[] run device code from host.
[] device memory usage in cuda-capable devices.
[] query sys-info on cuda-capable devices.

### A first program

- CPU/system memory -> **host**
- GPU/device memory -> **device**
- A function executes on the device -> **kernel**

```c++
__global__ void kernel () {
    return;
}
```

- empty function named **kernel()** that qualified with macro **`__global__`**
- call to the empty kernel function with `<<<1,1>>>`

- Using `__global__` to mind that the function should be compiled and execute on **device**.
- `<<<>>>` not passing the arguments to **device code**, but to **runtime**, determine the ways we launch the device.

#### Passing Parameters


By using the device to calculate simple add.

```c++
__global__ void add (int a, int b, int* c) {
    *c = a + b;
}

int main () {
    int c;
    int* device_c;
    HANDLE_ERROR (cudaMalloc ((void**)&device_c, sizeof (int)));

    add<<<1, 1>>> (2, 7, device_c);
    HANDLE_ERROR (cudaMemcpy (&c, device_c, sizeof (int), cudaMemcpyDeviceToHost));

    printf ("2 + 7 = %d\n", c);
    cudaFree (device_c);
    return 0;
}
```

- The target of the above program is *to calculate the sum of 2 params passed into the device, then return the answer to the host.*

1. The `device_c` is a pointer, and we should **allocate one memory area** on the device.
2. Pass the number into the `add` kernel function.
3. copy the contents from the device memory.

> Reminders:
> The Host can pass the pointer which use `cudaMalloc` to refrence a memory in device. However, **direct read or write** to this region is forbidden. We have the following restrictions:
>
> - Available: Host can **pass** the pointer to device or from device to host. **Device** can read or write through this pointer.
> - Unavailable: Host **CANNOT** use this pointer to read/write memory(since it is on the device).

So we have the following mapping from host <-> device:

- malloc -> cudaMalloc
- free -> cudaFree

To get the contents from the pointer that use `cudaMalloc`, we can use `cudaMemcpy` to replicate the contents to host's memory.

- Summary:
- Host pointer can R/W host mem. Device Pointer can R/W device mem.
- malloc -> cudaMalloc. free -> cudaFree.
- Using cudaMemcpy to get contents across devices. Macro `cudaMemcpyHostToDevice` copies the mem from host to device, while `cudaMemcpyDeviceToHost` dows the opposite.

#### Query Device

We can use function like this to get the **device properties**.

```c++
int main() {
    cudaDeviceProp properties;
    HANDLE_ERROR(cudaGetDeviceProperties(&properties, 0));
}
```

The structure `cudaDeviceProp` has all the informations inside. We displayed them below.

![img](https://img2024.cnblogs.com/blog/3143301/202510/3143301-20251012135057490-819262152.png)

Using this structure, we can get some basic infos.

```text
------ General Information for device 0 ------
Name:  NVIDIA GeForce RTX 4090
Compute capability:  8.9
Clock rate:  2520000
Device copy overlap:  Enabled
Kernel execution timeout :  Disabled
------ Memory Information for device 0 ------
Total global mem:  25250627584
Total constant Mem:  65536
Max mem pitch:  2147483647
Texture Alignment:  512
------ MP Information for device 0 ------
Multiprocessor count:  128
Shared mem per mp:  49152
Registers per mp:  65536
Threads in warp:  32
Max threads per block:  1024
Max thread dimensions:  (1024, 1024, 64)
Max grid dimensions:  (2147483647, 65535, 65535)
```

## 4 Parallel Program

### target

[] fundamental parallel
[] first parallel code

### Parallel Programming

In the very beginning I think that we should better get understanding of the architecture of executing the kernel function on GPU models.

#### The architecture of the GPU

Except *thread block clusters*(in Compute Capability 90 but I don't have the device of it), the 3 major components of the GPU are:

- thread
- block
- grid

![img](https://img2024.cnblogs.com/blog/3143301/202510/3143301-20251012171741404-1347335877.png)

Graph above shows the basic architecture of the cluster.

##### Dim3 structure

Both the gridDim and the blockDim can be described in structure called `dim3`. `dim3` describes the 3-dimensional distribution of the hierarchy model of the GPU software.

- for a dim3 block which has (Dx, Dy, Dz), the ID of the thread is (x + yDx + zDxDy). (x, y, z) can be acquired by `threadIdx.x`, `threadIdx.y`, `threadIdx.z`.
- The maximum thread size of a block can be get from previously described `cudaDeviceProp`. The `maxThreadsPerBlock` is set to 1024 in RTX 4090, which is currently we are learning with.

Also, we can have block being organized by dim3 inside a grid.

- for a dim3 grid which has (Dx, Dy, Dz), the id of a block with (x, y, z) is (x + yDx + zDxDy), which (x,y,z) is also can be acquired by `blockIdx.x`, `blockIdx.y`, `blockIdx.z`.
- Also, the properties can be known from `cudaDeviceProp`.

The geometric distribution of the dim3 is as follows:

![img](https://img2024.cnblogs.com/blog/3143301/202510/3143301-20251014152733103-976855493.png)

#### Write simple vecAdd using block or thread

First, we can write the code like this:

![img](https://img2024.cnblogs.com/blog/3143301/202510/3143301-20251013141717541-1297815268.png)

- First, we allocate 3 arrays on device.
- Then, we made some data, copy them from host to device.
- Then, the kernel function is set with a runtime `<<<N, 1>>>`, which means the grid has 1 dim N blocks, each block has 1 thread.
- after that, copy the mem from device to host, display and free them.

We can have a correct way. But this uses block as calculation.

To use thread, we can have:

```c++
// ...
__global__ void add (int* a, int* b, int* c) {
    int tid = threadIdx.x;
    // ...
}
int main () {
    // ...
    add<<<1, N>>> (dev_a, dev_b, dev_c);
    // ...
}
```

In this way, we launch with 1 block and N threads inside and calculate them in one block.

Since each block have limited threads parallelly execute the same code, we can extend to multiple block to execute this. Assuming N = 8192. We need 4 blocks, so we can have that `gridSize = N / threadPerBlock`. This should be round up when N is not integer times of `threadPerBlock`.

So we can have the following:

```c++
__global__ void add (int* a, int* b, int* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // ...
}
int main () {
    // ...
    // Kernel.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int gridSize = N / prop.maxThreadsPerBlock + 1;         // round up.
    add<<<gridSize, prop.maxThreadsPerBlock>>> (dev_a, dev_b, dev_c);
    // ...
}
```

#### A fun example

Julia Set. We have that: write a complex number $z$. Then we have:

$$
z_{n+1} = z_n^2 + C
$$

In this way, we can have a trial to draw the julia set in bitmap.

- part I: main function

What we need to do:

1. create 2 bitmaps, one on host, one on device.
2. allocate space for both map, using initializer for `CPUBitmap`, and use `cudaMalloc` for GPU bitmap.
3. kernel function execution~
4. copy the result from the device to host.
5. Use OpenCV to get the result image.

```c++
int main() {
    unsigned char *dev_bitmap;
    CPUBitmap bitmap (DIM, DIM, static_cast<void *>(dev_bitmap));
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_bitmap), bitmap.image_size()));
    
    dim3 gridDim (DIM, DIM);
    kernel<<<gridDim, 1>>> (dev_bitmap);
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_bitmap));

    cv::Mat image(DIM, DIM, CV_8UC4, bitmap.get_ptr());
    cv::imwrite("output.png", image);
    cv::imwrite("output.jpg", image);
}
```

We can define the other functions and structures like the following:

```c++
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
    int x = blockIdx.x, y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // Calculate julia value.
    int juliaVal = julia (x, y);

    // OpenCV with CV_8UC4 has the BGR\alpha image.
    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = 255 * juliaVal;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}
```

Some explanations:

- The Ptr will be represented as a RGBK image. In OpenCV, we have the image being interpreted as BGRK.
- The `__device__` function is different from the `__global__`. The former can be only executed in device, and be called by `__device__` or `__global__` decorated functions. The latter one shares between device and host.

In this example, we can have: (N = 5000)

![img](https://img2024.cnblogs.com/blog/3143301/202510/3143301-20251014142527978-1874736219.png)

## 5 Thread Cooperation

We have already known that:

- The architecture of the GPU software parallel model is: (From top -> bottom)
- grid -> (block clusters) -> block -> thread
- All the architectures can be organized by `dim3` structure.
- The calculation follows: `ind = x + y * Dx + z * Dy * Dx`.


We have the following kernel function show:

```c++
__global__ void add (int* a, int* b, int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
int main() {
    // ...
    constexpr int UNIT = 128;
    add<<<UNIT, UNIT>>> (dev_a, dev_b, dev_c);
    // ...
}
```

We have that: If we have dim3 for both block and grid, we will calculate for each `column` of thread. We have:

- Created a **runtime kernel** that has length of `gridDim.x * blockDim.x` threads.
- We need to map the index on the thread, that is, one by one.
- Hence we can think this as a **sliding window**. For the first round calculation, we map `[0, ..., UNIT * UNIT - 1]` elements on that.
- Then, keep the vector -> id map unchanged, slide to `[UNIT * UNIT, ..., 2 * UNIT * UNIT - 1]` elements.
- Since we can treat a sliding window as **parallel**, which **performs the same operation**, so we have each time move `tid <- tid + gridDim.x * blockDim.x`.

![img](https://img2024.cnblogs.com/blog/3143301/202510/3143301-20251019201403312-1704127181.png)

We have that:

```c++
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
```

For those offset may out of bound, we can use if statements.

### Shared Memory and Synchronization

- shared memory:
  - `__shared__` makes the variable resident in **shared memory**,  **create** a copy of this variable on each block.
  - thread **Unable** to **modify and see** the copy of this variable,   which is also seen in other block.
  - **shared memory** reside physically on GPU, but not residing in   off-chip DRAM.
  - Much **faster** than other typical buffers, can be seen as **cache**.
- communication:
  - Communicate between threads need **synchronization**.
  - When thread A write value to shared memory, which B needs, we should synchronize B after A complete.

#### Dot Product

We have the kernel function:

```c++
__global__ void dot (float* a, float* b, float* c) {
    // The buffer of shared memory.
    __shared__ float cache[threadsPerBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp = a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads ();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads ();
        i /= 2;
    }
    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}
```

We will divide them into parts to understand them.

**Part 1: Shared memory**:

We have used `__shared__ float cache[threadsPerBlock]` as the share memory. Here we have that `threadsPerBlock == blockDim.x`. We must use a constexpr value in order to create shared float memory.

**Part 2: Previous add**:

```c++
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < N) {
        temp = a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
```

Here we have that: Assuming we have a `blockDim.x` denote as `n`. So we have that for N elements:

![img](https://img2024.cnblogs.com/blog/3143301/202510/3143301-20251020145327566-1651653374.jpg)

So in shared, note `threadIdx.x` as t, `blockDim.x` as b, and `gridDim.x` as g, we have that the shared memory has:

```txt
[
    A(0) * B(0) + A(bg) * B(bg) + ..., 
    A(1) * B(1) + A(1 + bg) * B(1 + bg) + ...,
    ...,
    A(t) * B(t) + A(t + bg) * B(t + bg) + ...,
    ...
]
```

**After we complete** this operatiion, we need to sum these element. We can use **half-double** method to sum them, known as divide - and - conquer algorithm.

```c++
    __syncthreads ();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads ();
        i /= 2;
    }
    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
```

Inside the block, we divide them into 2 equal parts and sum them together.

![img](https://img2024.cnblogs.com/blog/3143301/202510/3143301-20251020150902072-235900418.png)

Multiple blocks will be reducted to the first element, where `threadIdx.x` is 0, that is, `cacheIndex` is 0.

In the end, in `main`, we sum them together to get the final result. To get the block numbers in grid, we need to use upper division: `(N + threadsPerBlock - 1) / threadsPerBlock)`. But we cannot use to less blocksPerGrid since the low utility, so we have `imin (32, (N + threadsPerBlock - 1) / threadsPerBlock);` as what we need.

**Synchronization**:

As we noticed above, we have three steps:

- product and sum the data to store in the shared mem.
- Reduction of the shared mem.

When we need to forcefully have a dependency, we need `__syncthreads()` to synchronize the threads. If we do not synchronize, their may be data races occurred. To detect data races, we can use `compute_sanitizer`.

```bash
compute-sanitizer --tool racecheck ./build/dot_product
```

When we enable all syncthreads, we have that:

```txt
========= COMPUTE-SANITIZER
Does GPU value 2.57229e+13 = 2.57236e+13?
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

Then, where will the data race happen? We have that the reduction should after the share_mem complete. This requires a **barrier**, thats how our `__syncthreads()` occurs.

After we remove the `__syncthreads()`, we have:

```txt
========= COMPUTE-SANITIZER
========= Error: Race reported between Read access at dot(float *, float *, float *)+0x230 in dot_product.cu:29
=========     and Write access at dot(float *, float *, float *)+0x180 in dot_product.cu:22 [69504 hazards]
=========     and Write access at dot(float *, float *, float *)+0x260 in dot_product.cu:29 [69248 hazards]
========= 
Does GPU value 1.90255e+13 = 2.57236e+13?
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
```

#### Wrong Optimization

```c++
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads ();
        i /= 2;
    }
```

Here we have that cacheIndex < i. We have a bold idea: Only wait and synchoronous those threads that need to **access shared mem** may accelerate ?

We try like this:

```c++
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
            __syncthreads ();
        }
        i /= 2;
    }
```

But this has a potential **thread_divergent** problem. That is, when threads that not evaluate the divergent branch, this thread **never** reach `__syncthreads ()`, so no threads can get through this barrier to start the next step.

