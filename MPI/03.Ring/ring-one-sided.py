from mpi4py import MPI
import numpy as np

rcv_buf = np.empty((), dtype=np.intc)

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

right = (my_rank+1) % size
left  = (my_rank-1+size) % size

sum = 0
snd_buf = np.array(my_rank, dtype=np.intc)

win = MPI.Win.Create(snd_buf, comm=comm_world)

for i in range(size):
  win.Fence()
  win.Get([rcv_buf, MPI.INT], left)
  win.Fence()
  sum += rcv_buf
  np.copyto(snd_buf, rcv_buf)

win.Free()

print(f"PE{my_rank}:\tSum = {sum}")