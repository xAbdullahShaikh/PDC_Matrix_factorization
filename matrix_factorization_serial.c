#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000 // Define a large matrix size

void initializeMatrix(double** matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (i == j) ? 1.0 : rand() % 100 + 1; // Initialize with random values and diagonal 1s.
        }
    }
}

void luDecomposition(double** matrix, double** L, double** U) {
    int i, j, k;

    // Initialize L to identity matrix and U to zero
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }

    // Perform LU Decomposition
    for (k = 0; k < N; k++) {
        // Compute U
        for (j = k; j < N; j++) {
            U[k][j] = matrix[k][j];
            for (i = 0; i < k; i++) {
                U[k][j] -= L[k][i] * U[i][j];
            }
        }

        // Compute L
        for (i = k + 1; i < N; i++) {
            L[i][k] = matrix[i][k];
            for (j = 0; j < k; j++) {
                L[i][k] -= L[i][j] * U[j][k];
            }
            L[i][k] /= U[k][k];
        }
    }
}

int main() {
    // Allocate memory for matrices
    double** matrix = (double*)malloc(N * sizeof(double));
    double** L = (double*)malloc(N * sizeof(double));
    double** U = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        matrix[i] = (double*)malloc(N * sizeof(double));
        L[i] = (double*)malloc(N * sizeof(double));
        U[i] = (double*)malloc(N * sizeof(double));
    }

    initializeMatrix(matrix);

    // Measure the time taken for LU decomposition
    clock_t start = clock();
    luDecomposition(matrix, L, U);
    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("LU decomposition of a %dx%d matrix completed in %.2f seconds.\n", N, N, time_taken);

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
        free(L[i]);
        free(U[i]);
    }
    free(matrix);
    free(L);
    free(U);

    return 0;
}

