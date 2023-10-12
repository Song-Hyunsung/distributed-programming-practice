#include <mpi.h>
#include <vector>

void copyTo(std::vector<int> &dst, std::vector<int> &src){
  for(int i = 0; i < src.size(); i++){
    dst.at(i) = src[i];
  }
}

int main(int argc, char* argv[]){
  int my_rank, num_procs, sum = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  std::vector<int> rcv_buf = { 0 };
  std::vector<int> snd_buf = { my_rank };
  int right = (my_rank+1) % num_procs;
  int left = (my_rank-1+num_procs) % num_procs;
  MPI_Request request;
  MPI_Status status;

  for(int i = 0; i < num_procs; i++){
    MPI_Irecv(&rcv_buf[0], 1, MPI_INT, left, 17, MPI_COMM_WORLD, &request);
    MPI_Send(&snd_buf[0], 1, MPI_INT, right, 17, MPI_COMM_WORLD);
    MPI_Wait(&request, &status);
    copyTo(snd_buf, rcv_buf);
    sum += rcv_buf[0];
  }

  printf("PE%d:\tSum = %d\n", my_rank, sum);

  MPI_Finalize();
}

