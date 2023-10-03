#include <mpi.h>

int main(int argc, char* argv[]){
    int my_rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if(num_procs == 2){
        if(my_rank == 0){
            printf("I am %i before send ping\n", my_rank);
            MPI_Send(NULL, 0, MPI_INT, 1, 17, MPI_COMM_WORLD);
        } else {
            MPI_Recv(NULL, 0, MPI_INT, 0, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("I am %i after recv ping\n", my_rank);
        }
    } else {
        // simple error message to not run the program when n != 2
        // this is because my_rank > 1 will be blocked with recv in the else statement
        if(my_rank == 0){
            printf("Number of processor is not 2, aborting.\n");
        }
    }

    MPI_Finalize();
}