from mpi4py import MPI

number_of_messages = 50
buffer = 0.0
status = MPI.Status()
comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()

if(num_procs == 2):
  start = MPI.Wtime();
  for i in range(number_of_messages):
    if (my_rank == 0):
      comm_world.send(buffer, dest=1, tag=17)
      buffer = comm_world.recv(source=1, tag=23, status=status)
    elif (my_rank == 1):
      buffer = comm_world.recv(source=0, tag=17, status=status)
      comm_world.send(buffer, dest=0, tag=23)
  finish = MPI.Wtime();

  if (my_rank == 0):
    msg_transfer_time = ((finish - start) / (2 * number_of_messages)) * 1e6
    print(f"Time for one message: {msg_transfer_time:f} micro seconds.")
else:
  if(my_rank == 0):
    print(f"Number of processor is not 2, aborting.")
