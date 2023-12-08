from mpi4py import MPI
import numpy as np
import random
import time

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()

# global_list_size = Total number of elements (evenly divisible by num_procs)
# global_sample_size = Sample size (evenly divisible by num_procs)
# num_procs = Number of processes
global_list_size = 10000
sub_sample_size = 16
global_sample_size = sub_sample_size * num_procs
global_sample = np.empty((global_sample_size), dtype=np.intc)
splitter = np.empty((num_procs+1), dtype=np.intc)
local_size = global_list_size // num_procs
local_list = np.empty((local_size), dtype=np.intc)
local_to_counts = np.zeros((num_procs), dtype=np.intc)
local_from_counts = np.zeros((num_procs), dtype=np.intc)
ex_local_to_counts = np.zeros((num_procs), dtype=np.intc)
ex_local_from_counts = np.zeros((num_procs), dtype=np.intc)

# Generate list only in root process
if(my_rank == 0):
  list = np.array([random.randint(0, 100000) for _ in range(global_list_size)], dtype=np.intc)

# Every process here now receives sublist with n/p elements
comm_world.Scatter(list, local_list, root=0)

# Measure start time
start_time = time.time()

# Every process should choose sub_sample_size subsample from its sublist
sub_sample = np.empty((sub_sample_size), dtype=np.intc)
chosen_sub_sample = set()
for i in range(sub_sample_size):
  random_number = local_list[random.randrange(local_size)]
  while(random_number in chosen_sub_sample):
    random_number = local_list[random.randrange(local_size)]
  sub_sample[i] = random_number
  chosen_sub_sample.add(random_number)

# Gather all the sub_samples onto the root process
comm_world.Gather(sub_sample, global_sample, root=0)

# Find splitters from global_sample
if(my_rank == 0):
  global_sample.sort()
  splitter[0] = -2147483648
  for i in range(1, num_procs):
    splitter[i] = ((global_sample[i-1] - global_sample[i]) // 2) + global_sample[i]
  splitter[num_procs] = 2147483647

# Broadcast splitter to all other processors
comm_world.Bcast(splitter, root=0)

# Sort each local list so that later alltoallv bucket distribution happens correctly
local_list.sort()

# For each local list, count how many elements are going to which processor
for num in local_list:
  for i in range(num_procs):
    if(num >= splitter[i] and num < splitter[i+1]):
      local_to_counts[i] += 1
      break

# Populate local_from_counts by using Alltoall collective
comm_world.Alltoall(local_to_counts, local_from_counts)

# Populate exclusive prefix sum for to/from counts
for i in range(1, num_procs):
  ex_local_to_counts[i] = ex_local_to_counts[i-1] + local_to_counts[i-1]
for i in range(1, num_procs):
  ex_local_from_counts[i] = ex_local_from_counts[i-1] + local_from_counts[i-1]

# Call alltoallv with exclusive counts to place numbers onto correct buckets
local_bucket_size = local_from_counts[num_procs-1] + ex_local_from_counts[num_procs-1]
local_bucket_list = np.zeros((local_bucket_size), dtype=np.intc)
comm_world.Alltoallv([local_list, local_to_counts, ex_local_to_counts, MPI.INT], [local_bucket_list, local_from_counts, ex_local_from_counts, MPI.INT])

# Now that all numbers are in correct bucket, sort local buckets
local_bucket_list.sort()

# Gather local bucket size to process 0
global_bucket_counts = np.zeros((num_procs), dtype=np.intc)
comm_world.Gather(local_bucket_size, global_bucket_counts, root=0)

# On root process, calculate exclusive prefix for bucket counts and Gather all sublists
ex_bucket_offsets = np.zeros((num_procs), dtype=np.intc)
if(my_rank == 0):
  for i in range(1, num_procs):
    ex_bucket_offsets[i] = ex_bucket_offsets[i-1] + global_bucket_counts[i-1]
comm_world.Gatherv([local_bucket_list, local_bucket_size, MPI.INT], [list, global_bucket_counts, ex_bucket_offsets, MPI.INT], root=0)

end_time = time.time()

if(my_rank == 0):
  sorted = True
  for i in range(len(list)-1):
    if list[i] > list[i+1]:
      print(f"{list[i]} VS. {list[i+1]}")
      sorted = False
      break
  if(not sorted):
    print(f"List is not sorted properly, errored out in above step.")
  else:
    print(f"List is sorted properly.")
    print(f"{list}")
    print(f"Total execution time: {end_time - start_time}")
