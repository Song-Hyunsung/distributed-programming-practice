from mpi4py import MPI
from random import randrange
import numpy as np
import sys

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()

# global_list_size = Total number of elements (evenly divisible by num_procs)
# global_sample_size = Sample size (evenly divisible by num_procs)
# num_procs = Number of processes
global_list_size = 12
global_sample_size = num_procs
global_sample = np.empty((global_sample_size), dtype=np.int_)
splitter = np.empty((num_procs+1), dtype=np.int_)
local_size = global_list_size // num_procs
sub_sample_size = num_procs // num_procs
local_list = np.empty((local_size), dtype=np.int_)

# Generate list only in root process
if(my_rank == 0):
  list = np.array([3,5,6,9,1,2,10,11,4,7,8,12])

# Every process here now receives sublist with n/p elements
comm_world.Scatter(list, local_list, root=0)

# Every process should choose sub_sample_size subsample from its sublist
sub_sample = np.empty((sub_sample_size), dtype=np.int_)
chosen_sub_sample = set()
for i in range(sub_sample_size):
  random_number = local_list[randrange(local_size)]
  while(random_number in chosen_sub_sample):
    random_number = local_list[randrange(local_size)]
  sub_sample[i] = random_number
  chosen_sub_sample.add(random_number)

# Gather all the sub_samples onto the root process
comm_world.Gather(sub_sample, global_sample, root=0)

# Find splitters from global_sample
if(my_rank == 0):
  # TODO THIS IS FOR DEV ONLY SINCE SMALL SAMPLE SIZE
  global_sample = [3,7,11]
  global_sample.sort()
  splitter[0] = -sys.maxsize-1
  for i in range(1, num_procs):
    splitter[i] = (global_sample[i-1] + global_sample[i]) // 2
  splitter[num_procs] = sys.maxsize

# Broadcast splitter to all other processors
comm_world.Bcast(splitter, root=0)

print(f"{my_rank}: {splitter}")