#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_rank_t rank, nprocs;
  gaspi_number_t seg_max;
  gaspi_segment_id_t s;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  //need the barrier to make sn is up
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  ASSERT(gaspi_segment_max(&seg_max));
  
  assert(seg_max == 32);

  for (s = 0; s < seg_max; s++)
    {
      ASSERT (gaspi_segment_create(s, 1024, GASPI_GROUP_ALL, GASPI_BLOCK,GASPI_MEM_UNINITIALIZED));
      //      ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
    }
  
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for (s = 0; s < seg_max; s++)
    ASSERT (gaspi_segment_delete(s));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
