#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_rank_t rank, nprocs;
  
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  //need the barrier to make sn is up
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  ASSERT (gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK,GASPI_MEM_UNINITIALIZED));
  ASSERT (gaspi_segment_create(0, _8MB, GASPI_GROUP_ALL, GASPI_BLOCK,GASPI_MEM_UNINITIALIZED));
  
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
