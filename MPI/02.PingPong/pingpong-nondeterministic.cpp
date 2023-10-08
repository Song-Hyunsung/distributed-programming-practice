#include <mpi.h>

int main(int argc, char* argv[]){
  int my_rank, num_procs;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if(num_procs == 2){
    if(my_rank == 0){
      printf("%i: before send ping\n", my_rank);
      MPI_Send(NULL, 0, MPI_INT, 1, 17, MPI_COMM_WORLD);
      MPI_Recv(NULL, 0, MPI_INT, 1, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("%i: after recv pong\n", my_rank);
    } else {
      MPI_Recv(NULL, 0, MPI_INT, 0, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("%i: after recv ping\n", my_rank);
      printf("%i: before send pong\n", my_rank);
      MPI_Send(NULL, 0, MPI_INT, 0, 23, MPI_COMM_WORLD);
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