from mpi4py import MPI

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()
buffer = [ None ]

if(num_procs == 2):
   if(my_rank == 0):
      print(f"I am {my_rank} before send ping")
      comm_world.send(None, dest=1, tag=17)
   else:
      result = comm_world.recv(source=0, tag=17)
      print(f"I am {my_rank} after  recv ping ")
else:
   # simple error message to not run the program when n != 2
   # this is because my_rank > 1 will be blocked with recv in the else statement
   if(my_rank == 0):
      print(f"Number of processor is not 2, aborting.")
