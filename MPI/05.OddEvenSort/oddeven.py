from mpi4py import MPI
import numpy as np

# this program is hard-coded to solve a specific problem with 4 cores
# can be modified to take input, set buffers, and distribute data dynamically
comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()
status = MPI.Status()
n = 16
local_n = n//num_procs
data = np.array([15, 11, 9, 16, 3, 14, 8, 7, 4, 6, 12, 10, 5, 2, 13, 1], dtype=np.intc)
# pre-allocate necessary buffers and arrays
rcv_buf = np.zeros(local_n, dtype=np.intc)
rcv_buf2 = np.zeros(local_n, dtype=np.intc)
temp = np.zeros(local_n, dtype=np.intc)

# scatter 16 data into 4 different processors
comm_world.Scatter((data, local_n, MPI.INT), (rcv_buf, local_n, MPI.INT))

print(f"Rank{my_rank}: Received data: ", end="")
for number in rcv_buf:
  print(number, end=" ")
print()

# sort each 4 separated local data
rcv_buf = np.sort(rcv_buf)

print(f"Rank{my_rank}: After local sort: ", end="")
for number in rcv_buf:
  print(number, end=" ")
print()

# define oddrank and evenrank partner for each processor
if my_rank % 2 == 0:
  oddrank = my_rank-1
  evenrank = my_rank+1
else:
  oddrank = my_rank+1
  evenrank = my_rank-1
# if at the edge, set to proc_null to skip during odd-even transposition
if oddrank == -1 or oddrank == num_procs:
  oddrank = MPI.PROC_NULL
if evenrank == -1 or evenrank == num_procs:
  evenrank = MPI.PROC_NULL

for p in range(num_procs):
  # on odd and even phase, send current local data to partner and receive partner's data on separate buffer
  if p % 2 == 0:
    comm_world.Sendrecv((rcv_buf, local_n, MPI.INT), evenrank, 1, (rcv_buf2, local_n, MPI.INT), evenrank, 1, status)
  else:
    comm_world.Sendrecv((rcv_buf, local_n, MPI.INT), oddrank, 1, (rcv_buf2, local_n, MPI.INT), oddrank, 1, status)
  # at this point
  # temp -> buffer holding "original" local data
  # rcv_buf2 -> buffer holding partner's local data
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
  
  print(f"Rank{my_rank}: At {p}th Iteration: ", end="")
  for number in rcv_buf:
    print(number, end=" ")
  print()

# gather all sorted local data onto process 0 into data array and print final result
comm_world.Gather((rcv_buf, local_n, MPI.INT), (data, local_n, MPI.INT), 0)
comm_world.Barrier()
if my_rank == 0:
  print("Sorted output:", end="")
  for number in data:
    print(number, end=" ")
  print()