#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000 // Define a large matrix size

void initializeMatrix(double* matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (i == j) ? 1.0 : rand() % 100 + 1;
        }
    }
}

void luDecomposition(double* matrix, double* L, double* U, int rank, int size) {
    int i, j, k;
    int rows_per_proc = N / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_proc;

    // Initialize L to identity and U to zero
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < N; j++) {
            L[i * N + j] = (i == j) ? 1.0 : 0.0;
            U[i * N + j] = 0.0;
        }
    }

    for (k = 0; k < N; k++) {
        // Broadcast the k-th row of U to all processes
        if (rank == k / rows_per_proc) {
            for (j = k; j < N; j++) {
                U[k * N + j] = matrix[k * N + j];
            }
        }
        MPI_Bcast(&U[k * N + k], N - k, MPI_DOUBLE, k / rows_per_proc, MPI_COMM_WORLD);

        // Compute L
        for (i = (k >= start_row) ? k + 1 : start_row; i < end_row; i++) {
            L[i * N + k] = matrix[i * N + k] / U[k * N + k];
            for (j = k + 1; j < N; j++) {
                matrix[i * N + j] -= L[i * N + k] * U[k * N + j];
            }
        }

        // Compute U
        for (j = k + 1; j < N; j++) {
            if (rank == k / rows_per_proc) {
                for (i = 0; i < k; i++) {
                    U[k * N + j] -= L[k * N + i] * U[i * N + j];
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double* matrix = (double*)malloc(N * N * sizeof(double));
    double* L = (double*)malloc(N * N * sizeof(double));
    double* U = (double*)malloc(N * N * sizeof(double));

    if (rank == 0) {
        initializeMatrix(matrix);
    }

    MPI_Bcast(matrix, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    luDecomposition(matrix, L, U, rank, size);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("LU decomposition of a %dx%d matrix completed in %.2f seconds.\n", N, N, end_time - start_time);
    }

    free(matrix);
    free(L);
    free(U);

    MPI_Finalize();
    return 0;
}
