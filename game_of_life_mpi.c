/* 
   Name: Michal Zajac
   Email: mpzajac@crimson.ua.edu  
   Course Section: CS 481  
   Homework #: 4 
   Instructions to compile the program: 
        mpicc -o game_of_life_mpi game_of_life_mpi.c
   Instructions to run the program: 
       Once compiled, run the program with the desired board size, max generations, number of processes, and output directory as arguments:
           mpiexec -n <num_processes> ./game_of_life_mpi <board_size> <max_generations> <num_processes> <output_dir>
       Example:
           mpiexec -n 4 ./game_of_life_mpi 5000 5000 4 /scratch/$USER
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>    // For MPI functions
#include <time.h>   // For timing
#include <unistd.h> // For getpid()

#define DEAD 0
#define ALIVE 1

// Function Declarations
void initialize_board(int* board, int size);
void initialize_local_board(int** local_board, int* local_board_data, int local_rows, int size);
void update_generation(int** current_gen, int** next_gen, int local_rows, int size);
int count_alive_neighbors(int** board, int row, int col);
int copy_and_check_equal(int** source, int** dest, int local_rows, int size);
void write_board_to_file(int *full_board_data, int size, const char *filename);
void print_board(int** board, int local_rows, int size);
void print_initial_board(int *board, int size);

int main(int argc, char *argv[]) {
    // MPI variables
    int comm_sz, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Check if the correct number of arguments is provided
    if (argc != 5) {
        if (my_rank == 0) {
            printf("Usage: %s <board_size> <max_generations> <num_processes> <output_dir>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int size = atoi(argv[1]);
    int max_generations = atoi(argv[2]);
    int num_processes = atoi(argv[3]);
    char* output_dir = argv[4];

    // Check if the number of processes matches
    if (comm_sz != num_processes) {
        if (my_rank == 0) {
            printf("Error: Number of processes specified (%d) does not match the number of MPI processes (%d).\n", num_processes, comm_sz);
        }
        MPI_Finalize();
        return 1;
    }

    // Compute the number of rows each process will handle
    int local_rows = size / comm_sz; // Base number of rows each process will handle
    int remainder = size % comm_sz; // Remainder rows to be distributed across processes

    // Allocate arrays for managing data distribution and collection
    int* sendcounts = malloc(comm_sz * sizeof(int)); // Number of elements each process will send
    int* displs = malloc(comm_sz * sizeof(int));     // Starting index for each process's data
    int* recvcounts = malloc(comm_sz * sizeof(int)); // Number of elements each process will receive
    int* recvdispls = malloc(comm_sz * sizeof(int)); // Receiving starting index for each process

    // Set up the send and receive counts and displacements for each process
    for (int i = 0; i < comm_sz; i++) {
        sendcounts[i] = (size / comm_sz) * size;
        if (i < remainder) {
            sendcounts[i] += size;
        }
        displs[i] = (i * (size / comm_sz) + (i < remainder ? i : remainder)) * size;
        recvcounts[i] = sendcounts[i];
        recvdispls[i] = displs[i];
    }

    local_rows = sendcounts[my_rank] / size;

    // Allocate local boards including ghost rows and columns
    int** current_gen = malloc((local_rows + 2) * sizeof(int*));
    int** next_gen = malloc((local_rows + 2) * sizeof(int*));
    for (int i = 0; i < local_rows + 2; i++) {
        current_gen[i] = malloc((size + 2) * sizeof(int));
        next_gen[i] = malloc((size + 2) * sizeof(int));
    }

    // Allocate memory for local board data (without ghost rows and columns)
    int* local_board_data = malloc(local_rows * size * sizeof(int));

    // Process 0 initializes the full board and scatters it
    if (my_rank == 0) {
        // Seed the random number generator with a fixed seed for reproducibility
        srand(12345);

        int* full_board_data = malloc(size * size * sizeof(int));
        initialize_board(full_board_data, size);

        // Uncomment the next line to print the initial board before scattering
        print_initial_board(full_board_data, size);

        // Scatter the board data to all processes
        MPI_Scatterv(full_board_data, sendcounts, displs, MPI_INT, local_board_data, sendcounts[my_rank], MPI_INT, 0, MPI_COMM_WORLD);

        free(full_board_data);
    } else {
        // Receive scattered data
        MPI_Scatterv(NULL, sendcounts, displs, MPI_INT, local_board_data, sendcounts[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Initialize local boards with ghost cells
    initialize_local_board(current_gen, local_board_data, local_rows, size);
    initialize_local_board(next_gen, NULL, local_rows, size); // Initialize ghost cells in next_gen

    free(local_board_data);

    // Start timing
    double start_time = MPI_Wtime();

    int global_changed = 1;
    int gen;
    for (gen = 0; gen < max_generations && global_changed; gen++) {
        // Exchange ghost rows with neighboring processes using MPI_Sendrecv
        int up_rank = my_rank - 1;
        int down_rank = my_rank + 1;
        if (up_rank < 0) up_rank = MPI_PROC_NULL;
        if (down_rank >= comm_sz) down_rank = MPI_PROC_NULL;

        // Send upper data row and receive lower ghost row
        MPI_Sendrecv(current_gen[1], size + 2, MPI_INT, up_rank, 0,
                     current_gen[local_rows + 1], size + 2, MPI_INT, down_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Send lower data row and receive upper ghost row
        MPI_Sendrecv(current_gen[local_rows], size + 2, MPI_INT, down_rank, 1,
                     current_gen[0], size + 2, MPI_INT, up_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Update the next generation based on the current generation
        update_generation(current_gen, next_gen, local_rows, size);

        // Check for changes and copy next_gen to current_gen
        int local_changed = !copy_and_check_equal(next_gen, current_gen, local_rows, size);

        // Check if any process had changes
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        // Synchronize processes before next iteration
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Stop timing
    double end_time = MPI_Wtime();
    double time_elapsed = end_time - start_time;

    if (my_rank == 0) {
        printf("Program exited after %d generations.\n", gen);
        printf("Execution Time: %.6f seconds\n", time_elapsed);
    }

    // Gather the final board data to process 0
    int* local_result_data = malloc(local_rows * size * sizeof(int));
    // Flatten current_gen into local_result_data
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= size; j++) {
            local_result_data[(i - 1) * size + (j - 1)] = current_gen[i][j];
        }
    }

    int* final_board_data = NULL;
    if (my_rank == 0) {
        final_board_data = malloc(size * size * sizeof(int));
    }

    MPI_Gatherv(local_result_data, local_rows * size, MPI_INT,
                final_board_data, recvcounts, recvdispls, MPI_INT,
                0, MPI_COMM_WORLD);

    // Write the final board to a file
    if (my_rank == 0) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/mpi_final_board.txt", output_dir);
        write_board_to_file(final_board_data, size, filename);
        free(final_board_data);
    }

    // Free allocated memory
    for (int i = 0; i < local_rows + 2; i++) {
        free(current_gen[i]);
        free(next_gen[i]);
    }
    free(current_gen);
    free(next_gen);
    free(local_result_data);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);

    MPI_Finalize();
    return 0;
}

// Function to initialize the full board (only on process 0)
void initialize_board(int* board, int size) {
    for (int i = 0; i < size * size; i++) {
        board[i] = rand() % 2;  // Randomly set cells as ALIVE or DEAD
    }
}

// Function to initialize the local boards with ghost cells
void initialize_local_board(int** local_board, int* local_board_data, int local_rows, int size) {
    // Initialize ghost cells to DEAD
    for (int i = 0; i < local_rows + 2; i++) {
        for (int j = 0; j < size + 2; j++) {
            local_board[i][j] = DEAD;
        }
    }

    // If local_board_data is provided, populate the local board
    if (local_board_data != NULL) {
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= size; j++) {
                local_board[i][j] = local_board_data[(i - 1) * size + (j - 1)];
            }
        }
    }
}

// Function to update the generation and return 1 if there are changes
void update_generation(int** current_gen, int** next_gen, int local_rows, int size) {
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= size; j++) {
            int alive_neighbors = count_alive_neighbors(current_gen, i, j);

            if (current_gen[i][j] == ALIVE) {
                if (alive_neighbors < 2 || alive_neighbors > 3) {
                    next_gen[i][j] = DEAD;  // Loneliness or overpopulation
                } else {
                    next_gen[i][j] = ALIVE;  // Survives
                }
            } else {
                if (alive_neighbors == 3) {
                    next_gen[i][j] = ALIVE;  // Reproduction
                } else {
                    next_gen[i][j] = DEAD;
                }
            }
        }
    }
}

// Function to count alive neighbors for a cell
int count_alive_neighbors(int** board, int row, int col) {
    int count = 0;
    // i and j range from -1 to 1, covering the 3x3 grid centered on (row, col).
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i != 0 || j != 0) { // Skip the current cell itself
                count += board[row + i][col + j];
            }
        }
    }
    return count;
}

// Copy the next generation to the current generation and check if they are identical
int copy_and_check_equal(int** source, int** dest, int local_rows, int size) {
    int identical = 1;  // Assume they are identical until proven otherwise
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= size; j++) {
            if (source[i][j] != dest[i][j]) {
                identical = 0;  // Found a difference, they are not identical
            }
            dest[i][j] = source[i][j];  // Copy the value from source to dest
        }
    }
    return identical;  // Return 1 if boards are identical, 0 otherwise
}

// Function to write the final board to a file
void write_board_to_file(int *board_data, int size, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size * size; i++) {
        fprintf(fp, "%d ", board_data[i]);
        if ((i + 1) % size == 0) {
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}

// Function to print the local board (for debugging)
void print_board(int** board, int local_rows, int size) {
    // Loop through only the actual board cells, excluding the ghost cells
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= size; j++) {
            printf("%d ", board[i][j]);  // Print 1 for ALIVE and 0 for DEAD
        }
        printf("\n");
    }
    printf("\n");
}

// Function to print the initial full board (only on process 0)
void print_initial_board(int *board, int size) {
    printf("Initial Board:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", board[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}
