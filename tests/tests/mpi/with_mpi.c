#include <stdio.h>

#include <mpi.h>
#include <GASPI.h>
#include <test_utils.h>


int main(int argc, char *argv[])
{
  gaspi_rank_t rank, nnodes;
  
  MPI_Init(&argc, &argv);

  ASSERT(gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_rank(&rank));
  ASSERT(gaspi_proc_num(&nnodes));

  printf("Hello GASPI rank %u of %u\n", rank, nnodes);
  
  gaspi_proc_term(GASPI_BLOCK);

  MPI_Finalize();

  return 0;
}
