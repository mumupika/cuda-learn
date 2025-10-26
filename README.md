# CUDA-LEARN

This is self-learning cuda from multiple resources. Currently, we have:

- cuda-by-example
- nccl

## Run

To Ensure running, we are required:

```txt
cmake >= 3.26
opencv4
opengl
freeglut
nccl
cuda-toolchain
cuda support compute capability of 8.9
```

The code was evaluated on 2 node with each has 2 L40 GPU.

## Build

```shell
sh cmake.sh
```
