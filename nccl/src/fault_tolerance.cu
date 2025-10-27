/*
    For fault tolerance.

    Asynchoronously submit task -> wait until timeout -> broadcast status -> restart.
*/

#include "comm.h"
#include "mpi.h"
#include "mpi_comm.h"
#include <chrono>
#include <nccl.h>

bool check_timeout (const std::chrono::time_point<std::chrono::steady_clock>& start,
                    int timeout) {
    const auto end = std::chrono::steady_clock::now ();
    if (std::chrono::duration_cast<std::chrono::seconds> (end - start) <
        std::chrono::seconds (timeout)) {
        return false;
    }
    return true;
}

ncclResult_t restartNCCL (ncclComm_t* comm, ncclUniqueId* id, int myRank, int nRanks) {
    // finalizing NCCL
    ncclCommDestroy (*comm);

    // Restart again.
    if (myRank == 0) {
        ncclGetUniqueId (id);
    }
    MPICHECK (MPI_Bcast ((void*)&id, sizeof (ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Trying to restart init again. If failed, report failed and exit.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclResult_t async_state = ncclInProgress;
    const auto start = std::chrono::steady_clock::now ();
    int timeout = 30;
    config.blocking = 0;
    NCCLCHECK (ncclGroupStart ());
    NCCLCHECK (ncclCommInitRankConfig (comm, nRanks, *id, myRank, &config));
    do {
        NCCLCHECK (ncclCommGetAsyncError (*comm, &async_state));
    } while (async_state == ncclInProgress && check_timeout (start, timeout) != true);
    NCCLCHECK (ncclGroupEnd ());
    if (check_timeout (start, timeout) == true || async_state != ncclSuccess) {
        ncclCommAbort (*comm);
        return async_state;
    }
    return async_state; // ncclSuccess can be return.
}

void recover_data (float* dev_s, float* dev_r, float* host, int size) {
    CUDACHECK (cudaMemset (dev_r, 0, sizeof (float) * size));
    CUDACHECK (cudaMemcpy (dev_s, host, sizeof (float) * size, cudaMemcpyHostToDevice));
}

void reportErrorGlobally (bool abortFlag, bool *globalFlag, int myRank) {
    abortFlag = true;
    *globalFlag = abortFlag;
    MPI_Bcast ((void*)globalFlag, sizeof (globalFlag), MPI_BYTE, myRank, MPI_COMM_WORLD);
}

void fault_tolearance_all_reduce (int myRank, int nRanks, int localRank) {
    int size = 32 * 1024 * 1024;

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff, *hostbuff;
    cudaStream_t s;

    // =================== ALLOCATE BASIC RESOURCES ===================

    // get NCCL unique ID at rank 0 and broadcast it to all others
    if (myRank == 0)
        ncclGetUniqueId (&id);
    MPICHECK (MPI_Bcast ((void*)&id, sizeof (id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK (cudaSetDevice (localRank));
    CUDACHECK (cudaMalloc (&sendbuff, size * sizeof (float)));
    CUDACHECK (cudaMalloc (&recvbuff, size * sizeof (float)));
    hostbuff = (float*)malloc (sizeof (float) * size);
    for (int i = 1.0; i < size; i++) {
        hostbuff[i] = float (i);
    }
    CUDACHECK (cudaMemcpy (sendbuff, hostbuff, sizeof (float) * size, cudaMemcpyHostToDevice));
    CUDACHECK (cudaStreamCreate (&s));

    // =================== Start Initialize comm ===================

    // Using clock and flag to record status of Initialization with fault tolerance.
    bool globalFlag = true;
    bool abortFlag = false;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclResult_t async_state = ncclInProgress;
    int timeout = 30;
    config.blocking = 0;

    // FIXME: One device per process can not use GroupStart/GroupEnd?
    NCCLCHECK (ncclGroupStart ());
    auto start = std::chrono::steady_clock::now ();
    NCCLCHECK (ncclCommInitRankConfig (&comm, nRanks, id, myRank, &config));
    do {
        NCCLCHECK (ncclCommGetAsyncError (comm, &async_state));
    } while (async_state == ncclInProgress && check_timeout (start, timeout) != true);
    NCCLCHECK (ncclGroupEnd ());

    // Broadcast to the whole processes. We need a barrier since here has a divergent.
    if (check_timeout (start, timeout) == true || async_state != ncclSuccess) {
        reportErrorGlobally (abortFlag, &globalFlag, myRank);
    }
    MPI_Barrier (MPI_COMM_WORLD);
    if (globalFlag == true) {
        ncclCommAbort (comm);
        // Free all resources and Renew again!
        NCCLCHECK (restartNCCL (&comm, &id, myRank, nRanks));
    }

    // =================== Start all reduce ===================

    globalFlag = true;
    abortFlag = false;
    NCCLCHECK (ncclGroupStart ());
    start = std::chrono::steady_clock::now ();
    NCCLCHECK (ncclAllReduce ((const void*)sendbuff, (void*)recvbuff, size,
                              ncclFloat, ncclSum, comm, s));
    NCCLCHECK (ncclGroupStart ());
    do {
        NCCLCHECK (ncclCommGetAsyncError (comm, &async_state));
    } while (async_state != ncclSuccess && check_timeout (start, timeout) != true);
    // Broadcast to the whole processes. We need a barrier since here has a divergent.
    if (check_timeout (start, timeout) == true || async_state != ncclSuccess) {
        reportErrorGlobally (abortFlag, &globalFlag, myRank);
    }
    MPI_Barrier (MPI_COMM_WORLD);
    if (globalFlag == true) {
        ncclCommAbort (comm);
        // Free all resources and Renew again!
        NCCLCHECK (restartNCCL (&comm, &id, myRank, nRanks));

        // Reset the data and Retry again.
        recover_data (sendbuff, recvbuff, hostbuff, size);
        globalFlag = true;
        abortFlag = false;
        NCCLCHECK (ncclGroupStart ());
        start = std::chrono::steady_clock::now ();
        NCCLCHECK (ncclAllReduce ((const void*)sendbuff, (void*)recvbuff, size,
                                  ncclFloat, ncclSum, comm, s));
        NCCLCHECK (ncclGroupStart ());
        do {
            NCCLCHECK (ncclCommGetAsyncError (comm, &async_state));
        } while (async_state != ncclSuccess && check_timeout (start, timeout) != true);
        NCCLCHECK (async_state); // Check the async_state and exit immediately if failed.
    }

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK (cudaStreamSynchronize (s));

    // free device buffers
    CUDACHECK(cudaMemcpy(hostbuff, recvbuff, sizeof(float) * size, cudaMemcpyDeviceToHost));
    CUDACHECK (cudaFree (sendbuff));
    CUDACHECK (cudaFree (recvbuff));

    // finalizing NCCL
    ncclCommDestroy (comm);

    // finalizing MPI
    MPICHECK (MPI_Finalize ());

    printf ("[MPI Rank %d] Success \n", myRank);
    return;
}

int main (int argc, char* argv[]) {

    int myRank, nRanks, localRank = 0;

    // initializing MPI
    MPICHECK (MPI_Init (&argc, &argv));
    MPICHECK (MPI_Comm_rank (MPI_COMM_WORLD, &myRank)); // MyRank -> stands the process rank in mpi.
    MPICHECK (MPI_Comm_size (MPI_COMM_WORLD, &nRanks)); // How many processes used in openmpi.

    // calculating localRank based on hostname which is used in selecting a GPU
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

    return 0;
}