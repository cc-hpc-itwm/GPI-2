#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

//alloc a segment, delete it and re-create it
int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t rank, nprocs, i;

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  ASSERT (gaspi_segment_create(0, 1024, 
			       GASPI_GROUP_ALL, 
			       GASPI_BLOCK, 
			       GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_segment_delete(0));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_segment_create(0, 2048,
  			       GASPI_GROUP_ALL,
  			       GASPI_BLOCK,
  			       GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_segment_delete(0));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
