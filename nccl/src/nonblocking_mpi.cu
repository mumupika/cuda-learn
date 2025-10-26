#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdint.h>

#include "comm.h"
#include "mpi_comm.h"

__global__ void fill_data (int dev_rank, int size, float* buff) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        buff[tid] = tid * size;
    }
    __syncthreads ();
}

void non_blocking_all_reduce (int myRank, int nRanks, int localRank) {
    // Set up the communication area and broadcast to all devices to host.
    ncclUniqueId id;
    if (myRank == 0) {
        NCCLCHECK (ncclGetUniqueId (&id));
    }
    MPICHECK (MPI_Bcast ((void*)&id, sizeof (ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

    int nDev = 2; // 1 Process -> 2 GPU

    float** sendbuff = (float**)malloc (nDev * sizeof (float*));
    float** recvbuff = (float**)malloc (nDev * sizeof (float*));
    cudaStream_t* s = (cudaStream_t*)malloc (nDev * sizeof (cudaStream_t));
    int size = 1024;

    for (int i = 0; i < nDev; i++) {
        CUDACHECK (cudaSetDevice (localRank * nDev + i));
        CUDACHECK (cudaMalloc ((void**)&sendbuff[i], sizeof (float) * size));
        CUDACHECK (cudaMalloc ((void**)&recvbuff[i], sizeof (float) * size));
        // fill in the data.
        int threadsPerBlock = 1024;
        int blockPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        CUDACHECK(cudaStreamCreate(&s[i]));
        fill_data<<<blockPerGrid, threadsPerBlock, 0, s[i]>>> (localRank * nDev + i,
                                                               size, sendbuff[i]);
        CUDACHECK (cudaMemset (recvbuff[i], 0, sizeof (float) * size));
    }

    // Synchronize until all stream completed...
    for (int i = 0; i < nDev; i++) {
        CUDACHECK (cudaStreamSynchronize (s[i]));
    }

    // Initializing NCCL ranks. Group initialize the comms.
    ncclComm_t* comms = (ncclComm_t *) malloc(sizeof(ncclComm_t) * nDev);
    ncclResult_t* state = (ncclResult_t *) malloc(sizeof(ncclResult_t) * nDev);
    ncclConfig_t* config = (ncclConfig_t *) malloc(sizeof(ncclConfig_t) * nDev);
    ncclResult_t ret = ncclInProgress;
    for(int i = 0; i < nDev; i++) {
        config[i] = NCCL_CONFIG_INITIALIZER;
        config[i].blocking = 0;
    }

    ncclGroupStart();
    for(int i = 0; i < nDev; i++) {
        ncclCommInitRankConfig(&comms[i], nDev, id, i, &config[i]);
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
        printf("[MPI Rank %d] Success: ", myRank);
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

int main (int argc, char* argv[]) {
    int myRank, nRanks, localRank = 0;

    // MPI initialization.
    MPICHECK (MPI_Init (&argc, &argv));
    MPICHECK (MPI_Comm_rank (MPI_COMM_WORLD, &myRank));
    MPICHECK (MPI_Comm_size (MPI_COMM_WORLD, &nRanks));

    // localRank calculation based on hostname.
    uint64_t* hostHashs = (uint64_t*)malloc (sizeof (uint64_t) * nRanks);
    char hostname[1024];
    getHostName (hostname, 1024);
    hostHashs[myRank] = getHostHash (hostname);
    MPICHECK (MPI_Allgather (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                             sizeof (uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++) {
        if (p == myRank)
            break;
        if (hostHashs[p] == hostHashs[myRank])
            localRank++;
    }

    non_blocking_all_reduce (myRank, nRanks, localRank);
}