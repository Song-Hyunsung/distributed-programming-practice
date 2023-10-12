from mpi4py import MPI
import numpy as np

rcv_buf = np.empty((), dtype=np.intc)
status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

right = (my_rank+1)      % size;
left  = (my_rank-1+size) % size;

sum = 0
snd_buf = np.array(my_rank, dtype=np.intc)

for i in range(size):
  request = comm_world.Isend((snd_buf, 1, MPI.INT), dest=right, tag=17)
  comm_world.Recv((rcv_buf, 1, MPI.INT), source=left, tag=17, status=status)
  request.Wait()
  np.copyto(snd_buf, rcv_buf)
  sum += rcv_buf
print(f"PE{my_rank}:\tSum = {sum}")