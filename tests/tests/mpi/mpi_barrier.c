#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <GASPI.h>
#include <test_utils.h>


int main(int argc, char *argv[])
{
  int i;
  
  int mpi_rank, mpi_sz, mpi_ret;
  gaspi_rank_t rank, nnodes;
  
  if(MPI_Init(&argc, &argv) != MPI_SUCCESS)
    return EXIT_FAILURE;
  
  ASSERT(gaspi_proc_init(GASPI_BLOCK));
  
  ASSERT(gaspi_proc_rank(&rank));
  ASSERT(gaspi_proc_num(&nnodes));
  
  mpi_ret = MPI_Comm_size(MPI_COMM_WORLD, &mpi_sz);
  assert(mpi_ret == 0);
  
  mpi_ret = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  assert(mpi_ret == 0);

  
  for(i = 0; i < 1000; i++)
    {
      ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
      assert(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
    }

  /* sync and finish */
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  assert(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);

  ASSERT(gaspi_proc_term(GASPI_BLOCK));
					 
  if(MPI_Finalize() != MPI_SUCCESS)
    return EXIT_FAILURE;

  return 0;
}
