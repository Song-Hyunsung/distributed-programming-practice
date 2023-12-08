from mpi4py import MPI
import numpy as np
import random
import time

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()
status = MPI.Status()

# performs parallel odd-even transposition sorting
def sort(data, n):
  local_n = n//num_procs
  # pre-allocate necessary buffers and arrays
  rcv_buf = np.zeros(local_n, dtype=np.intc)
  rcv_buf2 = np.zeros(local_n, dtype=np.intc)
  temp = np.zeros(local_n, dtype=np.intc)

  # scatter data into each processors
  comm_world.Scatter((data, local_n, MPI.INT), (rcv_buf, local_n, MPI.INT))

  rcv_buf = np.sort(rcv_buf)

  if my_rank % 2 == 0:
    oddrank = my_rank-1
    evenrank = my_rank+1
  else:
    oddrank = my_rank+1
    evenrank = my_rank-1

  if oddrank == -1 or oddrank == num_procs:
    oddrank = MPI.PROC_NULL
  if evenrank == -1 or evenrank == num_procs:
    evenrank = MPI.PROC_NULL

  for p in range(num_procs):
    if p % 2 == 0:
      comm_world.Sendrecv((rcv_buf, local_n, MPI.INT), evenrank, 1, (rcv_buf2, local_n, MPI.INT), evenrank, 1, status)
    else:
      comm_world.Sendrecv((rcv_buf, local_n, MPI.INT), oddrank, 1, (rcv_buf2, local_n, MPI.INT), oddrank, 1, status)
    for i in range(local_n):
      temp[i] = rcv_buf[i]
    
    if status.source == MPI.PROC_NULL:
      continue
    # start comparison for smaller rank, receive smaller number during comparison
    elif my_rank < status.source:
      i = j = k = 0
      while k < local_n:
        if j == local_n or (i < local_n and temp[i] < rcv_buf2[j]):
          rcv_buf[k] = temp[i]
          i += 1
        else:
          rcv_buf[k] = rcv_buf2[j]
          j += 1
        k += 1
    # start comparison for larger rank, receive larger number during comparison
    else:
      i = j = k = local_n - 1
      while k >= 0:
        if j == -1 or (i >= 0 and temp[i] >= rcv_buf2[j]):
          rcv_buf[k] = temp[i]
          i -= 1
        else:
          rcv_buf[k] = rcv_buf2[j]
          j -= 1
        k -= 1

  comm_world.Gather((rcv_buf, local_n, MPI.INT), (data, local_n, MPI.INT), 0)
  comm_world.Barrier()

  if my_rank == 0:
    return data

# global_list_size = Total number of elements (evenly divisible by num_procs)
# global_sample_size = Sample size (evenly divisible by num_procs)
# num_procs = Number of processes
global_list_size = 10000
sub_sample_size = 64
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

global_sample = sort(global_sample, global_sample_size)

# Find splitters from global_sample
if(my_rank == 0):
  splitter[0] = -2147483648
  for i in range(1, num_procs):
    splitter[i] = ((global_sample[i-1] - global_sample[i]) // 2) + global_sample[i]
  splitter[num_procs] = 2147483647

# Broadcast splitter to all other processors
comm_world.Bcast(splitter, root=0)

# Perform butterfly operation instead of Alltoallv operation
bitmask = num_procs >> 1
which_splitter = bitmask
local_bucket_list = local_list

while(bitmask >= 1):
  partner = my_rank ^ bitmask
  snd_list = []
  tmp_list = []
  snd_count = 0
  if(my_rank < partner):
    local_splitter = splitter[which_splitter]
    for num in local_bucket_list:
      if(num >= local_splitter):
        snd_list.append(num)
        snd_count += 1
      else:
        tmp_list.append(num)
    bitmask >>= 1
    which_splitter -= bitmask
  else:
    local_splitter = splitter[which_splitter]
    for num in local_bucket_list:
      if(num >= local_splitter):
        tmp_list.append(num)
      else:
        snd_list.append(num)
        snd_count += 1
    bitmask >>= 1
    which_splitter += bitmask
  snd_buf = np.array((snd_list), dtype=np.intc)
  request = comm_world.Isend((snd_buf, snd_count, MPI.INT), dest=partner, tag=17)
  comm_world.Probe(partner, 17, status)
  rcv_count = status.Get_count(MPI.INT)
  rcv_buf = np.empty((rcv_count), dtype=np.intc)
  comm_world.Recv(rcv_buf, source=partner, tag=17)
  request.Wait()
  tmp_buf = np.array((tmp_list), dtype=np.intc)
  local_bucket_list = np.append(tmp_buf, rcv_buf)

# Now that all numbers are in correct bucket, sort local buckets
local_bucket_list.sort()

# Gather local bucket size to process 0
global_bucket_counts = np.zeros((num_procs), dtype=np.intc)
local_bucket_size = np.array((local_bucket_list.size), dtype=np.intc)
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
