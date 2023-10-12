from mpi4py import MPI
import numpy as np

sum = 0
comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

rcv_buf = np.empty((), dtype=np.intc)
snd_buf = np.array(my_rank, dtype=np.intc)

comm_world.Allreduce((snd_buf, 1, MPI.INT), (rcv_buf, 1, MPI.INT), MPI.SUM)
sum = rcv_buf

print(f"PE{my_rank}:\tSum = {sum}")