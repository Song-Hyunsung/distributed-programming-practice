#include <mpi.h>
#include <vector>

int main(int argc, char* argv[]){
  int my_rank, num_procs, sum = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  std::vector<int> rcv_buf = { 0 };
  std::vector<int> snd_buf = { my_rank };
  
  MPI_Allreduce(&snd_buf[0], &rcv_buf[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  sum = rcv_buf[0];

  printf("PE%d:\tSum = %d\n", my_rank, sum);

  MPI_Finalize();
}

