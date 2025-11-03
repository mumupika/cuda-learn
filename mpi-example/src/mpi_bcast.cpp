#include "mpi.h"
#include "mpi_comm.h"

// The test for MPI_Bcast.
// Run with: mpirun -n 2 ./build/mpi-example/mpi_bcast

int main (int argc, char* argv[]) {
    int rank, size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank); // Current rank in the process.
    MPI_Comm_size (MPI_COMM_WORLD, &size); // How many ranks.

    int* array = (int*)malloc (sizeof (int) * 10);
    memset(array, 0, sizeof(int) * 10);
    if (rank == 0) {
        for (int i = 0; i < 10; i++) {
            array[i] = i;
        }
    }
    MPI_Barrier (MPI_COMM_WORLD);

    if (rank == 0) {
        printf ("Before Bcast: \n [Rank 0]: \n");
        for (int i = 0; i < 10; i++) {
            printf ("%d ", array[i]);
        }
        printf ("\n");
    }
    MPI_Barrier (MPI_COMM_WORLD);

    if (rank == 1) {
        printf ("Before Bcast: \n [Rank 1]: \n");
        for (int i = 0; i < 10; i++) {
            printf ("%d ", array[i]);
        }
        printf ("\n");
    }
    MPI_Barrier (MPI_COMM_WORLD);

    MPI_Bcast (array, sizeof (int) * 10, MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf ("After Bcast: \n [Rank 0]: \n");
        for (int i = 0; i < 10; i++) {
            printf ("%d ", array[i]);
        }
        printf ("\n");
    }
    MPI_Barrier (MPI_COMM_WORLD);

    if (rank == 1) {
        printf ("After Bcast: \n [Rank 1]: \n");
        for (int i = 0; i < 10; i++) {
            printf ("%d ", array[i]);
        }
        printf ("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    printf("All completed with [rank %d]\n", rank);         // No barrier so this line is out of order.
    MPI_Finalize();
    return 0;
}