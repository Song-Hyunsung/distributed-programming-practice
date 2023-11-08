from mpi4py import MPI
import numpy as np

def Fc(x):
  return x * x

def Trap(left_pt, right_pt, trap_count, base_length):
  estimate = 0.0
  # calculate left and right boundary
  estimate = (Fc(left_pt) + Fc(right_pt)) / 2
  for i in range(1, trap_count):
    estimate += Fc(left_pt + i * base_length)
  estimate = estimate * base_length
  return estimate

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()
n = None
a = 0.0
b = 1.0

if my_rank == 0:
  n = int(input("Enter the value of n: "))
n = comm_world.bcast(n, root=0)

h = (b-a)/n
local_n = int(n/num_procs)
local_a = a + my_rank * local_n * h
local_b = local_a + local_n * h
local_int = Trap(local_a, local_b, local_n, h)

if my_rank != 0:
  snd_buf = np.array(local_int, dtype=np.double)
  comm_world.Send((snd_buf, 1, MPI.DOUBLE), dest=0, tag=0)
else:
  total_int = local_int
  rcv_buf = np.empty((), dtype=np.double)
  for i in range(1, num_procs):
    comm_world.Recv((rcv_buf, 1, MPI.DOUBLE), source=i, tag=0, status=None)
    total_int += rcv_buf

if my_rank == 0:
  print(f"With n = {n} trapezoids and {num_procs} processors, Integral of x^2 from {a} to {b} = {total_int}")
  print(f"Absolute error against theoretical value is: {abs((1/3) - total_int)}")