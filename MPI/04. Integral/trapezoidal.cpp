#include <mpi.h>

// integral of x^2
double Fc(double x){
  return x * x;
}

double Trap(double left_pt, double right_pt, int trap_count, double base_length){
  double estimate = 0.0;
  // calculate left and right boundary
  estimate = (Fc(left_pt) + Fc(right_pt)) / 2;
  for(int i = 1; i < trap_count; i++){
    estimate += Fc(left_pt + i * base_length);
  }
  estimate = estimate * base_length;
  return estimate;
}

int main(int argc, char* argv[]){
  // this program expects n to be evenly divisible by num_procs
  int my_rank, num_procs, n, local_n;
  double a = 0.0, b = 1.0, h, local_a, local_b;
  double local_int, total_int;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if(my_rank == 0){
    printf("Enter the value of n: \n");
    scanf("%d",&n);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // h (equal size interval) is same for all processes
  h = (b-a)/n;
  // number of trapezoids for each processors
  local_n = n/num_procs;
  // specify local start (a) and end (b)
  local_a = a + my_rank * local_n * h;
  local_b = local_a + local_n * h;
  // calculate local integral using trapezoidal method
  local_int = Trap(local_a, local_b, local_n, h);

  if(my_rank != 0){
    MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  } else {
    total_int = local_int;
    for(int source = 1; source < num_procs; source++){
      MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      total_int += local_int;
    }
  }

  if(my_rank == 0){
    printf("With n = %d trapezoids and %d processors, Integral of x^2 from %f to %f = %f\n", n, num_procs, a, b, total_int);
  }

  MPI_Finalize();
}

