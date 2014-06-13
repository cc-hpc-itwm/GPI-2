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
  
  printf("Hello GASPI rank %u of %u and MPI rank %d of %d\n",
	 rank, nnodes, mpi_rank, mpi_sz);

  const gaspi_size_t seg_size = (2 << 20);
  ASSERT(gaspi_segment_create(0, seg_size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
  gaspi_pointer_t seg_ptr;
  ASSERT(gaspi_segment_ptr(0, &seg_ptr));
  int * int_ptr = (int *) seg_ptr;

  int val_send = mpi_rank;
  MPI_Request request[mpi_sz];

  /* Communication: send buffer to GPI segment with MPI */
  for( i = 0; i < mpi_sz; i++)
    {
      assert(MPI_Isend(&val_send, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &(request[i])) == MPI_SUCCESS);

      /* GPI concurrent comm */
      ASSERT(gaspi_write(0, seg_size / 2, i, 0, seg_size / 2, sizeof(int), 0, GASPI_BLOCK));
    }

  for( i = 0; i < mpi_sz; i++)
    {
      assert(MPI_Recv(int_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS);
      assert(*int_ptr == i);
    }

  /* Wait Comm */
  assert(MPI_Waitall(mpi_sz, request, MPI_STATUSES_IGNORE) == MPI_SUCCESS);
  ASSERT(gaspi_wait(0, GASPI_BLOCK));
  
  /* sync and finish */
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  assert(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);

  ASSERT(gaspi_proc_term(GASPI_BLOCK));
					 
  if(MPI_Finalize() != MPI_SUCCESS)
    return EXIT_FAILURE;

  return 0;
}
