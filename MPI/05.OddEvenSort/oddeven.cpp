#include <mpi.h>
#include <algorithm>

// this program is hard-coded to solve a specific problem with 4 cores
// can obviously be modified to take input, set buffers, and distribute data dynamically
int main(int argc, char *argv[]){
  int my_rank, num_procs, local_n;
  int n = 16;
  int data[16] = {15, 11, 9, 16, 3, 14, 8, 7, 4, 6, 12, 10, 5, 2, 13, 1};
  int rcv_buf[4], rcv_buf2[4], temp[4];
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  local_n = n/num_procs;

  // scatter 16 data into 4 different processors
  MPI_Scatter(data, local_n, MPI_INT, &rcv_buf, 4, MPI_INT, 0, MPI_COMM_WORLD);

  printf("Rank%d: Received data:", my_rank);
  for (int i = 0; i < local_n; i++){
    printf("%d ", rcv_buf[i]);
  }
  printf("\n");

  // sort each 4 separated local data
  std::sort(rcv_buf, rcv_buf + local_n);

  printf("Rank%d: After local sort:", my_rank);
  for (int i = 0; i < local_n; i++){
    printf("%d ", rcv_buf[i]);
  }
  printf("\n");

  // define oddrank and evenrank for each processor
  // if processor is at edge, set it as MPI_PROC_NULL to ignore during sorting
  int oddrank, evenrank;
  if(my_rank % 2 == 0){
    oddrank = my_rank - 1;
    evenrank = my_rank + 1;
  } else {
    oddrank = my_rank + 1;
    evenrank = my_rank - 1;
  }
  if (oddrank == -1 || oddrank == num_procs){
    oddrank = MPI_PROC_NULL;
  }
  if (evenrank == -1 || evenrank == num_procs){
    evenrank = MPI_PROC_NULL;
  }

  for (int p = 0; p < num_procs; p++){
    // on odd and even phase, send current local data to partner and receive partner's data on separate buffer
    if (p % 2 == 1){
      MPI_Sendrecv(rcv_buf, local_n, MPI_INT, oddrank, 1, rcv_buf2, local_n, MPI_INT, oddrank, 1, MPI_COMM_WORLD, &status);
    } else {
      MPI_Sendrecv(rcv_buf, local_n, MPI_INT, evenrank, 1, rcv_buf2, local_n, MPI_INT, evenrank, 1, MPI_COMM_WORLD, &status);
    }
    // store current local data on temporary buffer
    // at this point
    // temp -> buffer holding "original" local data
    // rcv_buf2 -> buffer holding partner's local data
    for (int i = 0; i < local_n; i++){
      temp[i] = rcv_buf[i];
    }

    if(status.MPI_SOURCE == MPI_PROC_NULL){
      continue;
    // on smaller rank, receive smaller numbers during comparison
    } else if(my_rank < status.MPI_SOURCE){
      int i, j, k;
      for(i = j = k = 0; k < local_n; k++){
        if(j == local_n || (i < local_n && temp[i] < rcv_buf2[j])){
          rcv_buf[k] = temp[i++];
        } else {
          rcv_buf[k] = rcv_buf2[j++];
        }
      }
    // on larger rank, receive larger numbers during comparison
    } else {
      int i, j, k;
      for(int i = j = k = local_n - 1; k >= 0; k--){
        if (j == -1 || (i >= 0 && temp[i] >= rcv_buf2[j])){
          rcv_buf[k] = temp[i--];
        } else {
          rcv_buf[k] = rcv_buf2[j--];
        }
      }
    }

    printf("Rank%d: At %dth Iteration:", my_rank, p);
    for (int i = 0; i < local_n; i++){
      printf("%d ", rcv_buf[i]);
    }
    printf("\n");
  }

  // gather all sorted local data onto process 0 to print final result
  MPI_Gather(rcv_buf, local_n, MPI_INT, data, local_n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0){
    printf("Sorted output:");
    for (int i = 0; i < n; i++){
      printf("%d ", data[i]);
    }
    printf("\n");
  }

  MPI_Finalize();
}