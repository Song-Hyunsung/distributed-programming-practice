#include <mpi.h>

int main(int argc, char* argv[]){
  int my_rank, num_procs, buffer;
  const int number_of_messages = 50;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if(num_procs == 2){
    double start = MPI_Wtime();
    for(int i = 0; i < number_of_messages; i++){
      if(my_rank == 0){
        MPI_Send(&buffer, 1, MPI_INT, 1, 17, MPI_COMM_WORLD);
        MPI_Recv(&buffer, 1, MPI_INT, 1, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&buffer, 1, MPI_INT, 0, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&buffer, 1, MPI_INT, 0, 23, MPI_COMM_WORLD);
      }
    }
    double finish = MPI_Wtime();

    if(my_rank == 0){
      double msg_transfer_time = ((finish - start) / (2 * number_of_messages)) * 1e6;
      printf("Time for one message: %f micro seconds.\n", msg_transfer_time);
    }
  } else {
    if(my_rank == 0){
      printf("Number of processor is not 2, aborting.\n");
    }
  }

  MPI_Finalize();
}