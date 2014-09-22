#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

/* All only some nodes (rank < nprocs - 3) to a group and do a barrier */

int main(int argc, char *argv[])
{

  gaspi_group_t g;
  gaspi_rank_t gsize, nprocs, myrank;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT(gaspi_proc_rank(&myrank));

  ASSERT (gaspi_group_create(&g));
  ASSERT(gaspi_group_size(g, &gsize));
  assert((gsize == 0));

  if((nprocs > 3) && (myrank < nprocs / 2))
    {
      gaspi_rank_t i;
      gaspi_group_t gaspi_group_com;

      ASSERT(gaspi_group_create(&gaspi_group_com));

      for(i = 0; i < nprocs / 2; i++)
	ASSERT(gaspi_group_add(gaspi_group_com, i));

      ASSERT(gaspi_group_commit(gaspi_group_com, GASPI_BLOCK));
      
      ASSERT (gaspi_barrier(gaspi_group_com, GASPI_BLOCK));
    }
  
  //all barrier                                                                                           
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));
  return EXIT_SUCCESS;
}

