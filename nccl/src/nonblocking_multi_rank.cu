#include "comm.h"
#include <nccl.h>

// The CUDA kernel fill the data with {0,1,2,...};
__global__ void fill_data(float * sendbuff, int size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < size) {
        sendbuff[tid] = float(tid);
    }
    __syncthreads();
}

void non_blocking_multirank_allreduce(int nRanks) {
    int size = 1024 * 1024;

    // single process single thread for multiple devices.
    int nDev = nRanks;
    int *devs = (int *) malloc(sizeof(int) * nDev);
    for(int i = 0; i < nDev; i++) {
        devs[i] = i;
    }

    // Allocating the buffer and generating the streams.
    float** sendbuff = (float **) malloc(nDev * sizeof(float *));
    float** recvbuff = (float **) malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *) malloc(nDev * sizeof(cudaStream_t));

    // Fill the data inside.
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(devs[i]));
        CUDACHECK(cudaMalloc((void **)(&sendbuff[i]), size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)(&recvbuff[i]), size * sizeof(float)));
        int threadsPerBlock = 1024;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        fill_data<<<blocksPerGrid, threadsPerBlock>>> (sendbuff[i], size);
        CUDACHECK(cudaMemset(recvbuff[i], 0, sizeof(float) * size));
        CUDACHECK(cudaStreamCreate(s+i));
    }

    // Ensure all operations.
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(devs[i]));
        CUDACHECK(cudaDeviceSynchronize());
    }

    // Initializing NCCL ranks. Group initialize the comms.
    ncclComm_t* comms = (ncclComm_t *) malloc(sizeof(ncclComm_t) * nDev);
    ncclResult_t* state = (ncclResult_t *) malloc(sizeof(ncclResult_t) * nDev);
    ncclConfig_t* config = (ncclConfig_t *) malloc(sizeof(ncclConfig_t) * nDev);
    for(int i = 0; i < nDev; i++) {
        config[i] = NCCL_CONFIG_INITIALIZER;
        config[i].blocking = 0;
    }

    // Prepare commID.
    ncclResult_t ret;
    ncclUniqueId uniqueIds;
    NCCLCHECK(ncclGetUniqueId(&uniqueIds));

    ncclGroupStart();
    for(int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(devs[i]));
        ncclCommInitRankConfig(&comms[i], nDev, uniqueIds, i, &config[i]);
    }
    ret = ncclGroupEnd();
    if(ret == ncclInProgress) {
        for(int i = 0; i < nDev; i++) {
            // Wait until one complete. Then another.
            do {
                NCCLCHECK(ncclCommGetAsyncError(comms[i], &state[i]));
            } while(state[i] == ncclInProgress);
            NCCLCHECK(state[i]);
        }
    } else if (ret == ncclSuccess) {
        printf("NCCL kernel issue succeeded\n");
    } else {
        // Error occurred.
        NCCLCHECK(ret);
    }

    // NONBLOCKING allReduce.
    ncclGroupStart();
    for (int i = 0; i < nDev; i++) {
        ncclAllReduce(reinterpret_cast<const char *>(sendbuff[i]), static_cast<void *>(recvbuff[i]), size, ncclFloat, ncclSum, comms[i], s[i]);
    }
    ret = ncclGroupEnd();
    if(ret == ncclInProgress) {
        for(int i = 0; i < nDev; i++) {
            // Wait until one complete. Then another.
            do {
                NCCLCHECK(ncclCommGetAsyncError(comms[i], &state[i]));
            } while(state[i] == ncclInProgress);
            NCCLCHECK(state[i]);
        }
    } else if (ret == ncclSuccess) {
        printf("NCCL kernel issue succeeded\n");
    } else {
        // Error occurred.
        NCCLCHECK(ret);
    }

    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(devs[i]));
        cudaStreamSynchronize(s[i]);
    }

    // Get the data of the final result.
    float **hostData = (float **) malloc(nDev * sizeof(float *));
    for(int i = 0; i < nDev; i++) {
        hostData[i] = (float *) malloc(size * sizeof(float));
    }
    for(int i = 0; i < nDev; i++) {
        cudaMemcpy(hostData[i], recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    for(int i = 0; i < nDev; i++) {
        printf("Currently get GPU %d\n", i);
        for(int j = 0; j < 20; j++) {
            printf("%f ", hostData[i][j]);
        }
        printf("\n");
    }

    // Free all resources.
    for(int i = 0; i < nDev; i++) {
        ncclCommDestroy(comms[i]);
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        free(hostData[i]);
    }
    free(sendbuff);
    free(recvbuff);
    free(hostData);
}   
int main(int argc, char *argv[]) {
    // Set the nranks here.
    env_init(argc, argv);
    int nRanks = my_nranks;
    int nGPU;
    cudaGetDeviceCount(&nGPU);
    printf("Has %d devices\n", nGPU);

    non_blocking_multirank_allreduce(nRanks);
    return 0;
}