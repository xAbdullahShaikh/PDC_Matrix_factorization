#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h> // For sleep function
#include <pthread.h> // For threading

#define N 10000 // Define a large matrix size

// Structure to pass arguments to the timer thread
typedef struct {
    int running;
} TimerArgs;

void* displayTimer(void* args) {
    TimerArgs* timerArgs = (TimerArgs*)args;
    double start_time = omp_get_wtime();
   
    while (timerArgs->running) {
        double elapsed = omp_get_wtime() - start_time;
        printf("\rElapsed time: %.2f seconds", elapsed);
        fflush(stdout); // Ensure the output is displayed immediately
        sleep(1); // Update every second
    }

    return NULL;
}
void initializeMatrix(double** matrix) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (i == j) ? 1.0 : rand() % 100 + 1; // Initialize with random values and diagonal 1s.
        }
    }
}

void luDecomposition(double** matrix, double** L, double** U) {
    int i, j, k;

    // Initialize L to identity matrix and U to zero
    #pragma omp parallel for private(j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }

    // Perform LU Decomposition
    for (k = 0; k < N; k++) {
        // Compute U (parallelize outer loop)
        #pragma omp parallel for private(i)
        for (j = k; j < N; j++) {
            U[k][j] = matrix[k][j];
            for (i = 0; i < k; i++) {
                U[k][j] -= L[k][i] * U[i][j];
            }
        }

        // Compute L (parallelize outer loop)
        #pragma omp parallel for private(j)
        for (i = k + 1; i < N; i++) {
            L[i][k] = matrix[i][k];
            for (j = 0; j < k; j++) {
                L[i][k] -= L[i][j] * U[j][k];
            }
            L[i][k] /= U[k][k];
        }
    }
}

int main(int argc, char* argv[]) {
    // Check if the number of threads is provided
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse the number of threads from the command line
    int num_threads = atoi(argv[1]);
    if (num_threads <= 0) {
        fprintf(stderr, "Number of threads must be greater than 0.\n");
        return EXIT_FAILURE;
    }

    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    printf("Using %d threads for OpenMP parallel regions.\n", num_threads);

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

    // Timer thread setup
    TimerArgs timerArgs = {1}; // Set running to true
    pthread_t timerThread;
    pthread_create(&timerThread, NULL, displayTimer, (void*)&timerArgs);

    // Measure the time taken for LU decomposition
    double start = omp_get_wtime();
    luDecomposition(matrix, L, U);
    double end = omp_get_wtime();

    // Stop the timer
    timerArgs.running = 0;
    pthread_join(timerThread, NULL); // Wait for the timer thread to finish

    double time_taken = end - start;
    printf("\nLU decomposition of a %dx%d matrix completed in %.2f seconds.\n", N, N, time_taken);

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

