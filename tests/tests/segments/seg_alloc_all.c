#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

//alloc max number of segs, fail after that
//and them register them all with all nodes
//then delete them
int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  //need the barrier to make sn is up
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_rank_t rank, nprocs, i;
  gaspi_number_t seg_max;
  gaspi_segment_id_t s;

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  ASSERT(gaspi_segment_max(&seg_max));
  
  assert(seg_max == GASPI_MAX_MSEGS);

  for (s = 0; s < seg_max; s++)
    ASSERT (gaspi_segment_alloc(s, 1024, GASPI_MEM_INITIALIZED));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  EXPECT_FAIL (gaspi_segment_alloc(s, 1024, GASPI_MEM_INITIALIZED));
  
  for (i = 0; i < nprocs; i++)
    {
      gaspi_printf("register with %u\n", i);
      
      if(i == rank)
	continue;
      for (s = 0; s < seg_max; s++)
	ASSERT( gaspi_segment_register(s, i, GASPI_BLOCK));

      //      sleep(1);
    }

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for (s = 0; s < seg_max; s++)
    ASSERT (gaspi_segment_delete(s));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
