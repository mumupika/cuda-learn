#include "mpi.h"
#include "mpi_comm.h"

void reportGlobalError (bool abortFlag, bool* globalFlag) {
    MPI_Allreduce ((const void*)&abortFlag, (void*)globalFlag, sizeof (bool),
                   MPI_BYTE, MPI_LAND, MPI_COMM_WORLD);
}
int main (int argc, char* argv[]) {
    int rank, size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank); // Current rank in the process.
    MPI_Comm_size (MPI_COMM_WORLD, &size); // How many ranks.

    bool abortFlag = true, globalFlag = true;
    if (rank == 0) {
        abortFlag = false;
    }
    MPI_Barrier (MPI_COMM_WORLD);

    reportGlobalError (abortFlag, &globalFlag);
    // MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf ("[rank 0] abortFlag: %s, globalFlag: %s\n", abortFlag == true ? "true" : "false",
                globalFlag == true ? "true" : "false");
    }
    MPI_Barrier (MPI_COMM_WORLD);
    if (rank == 1) {
        printf ("[rank 1] abortFlag: %s, globalFlag: %s\n", abortFlag == true ? "true" : "false",
                globalFlag == true ? "true" : "false");
    }
    MPI_Barrier (MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}