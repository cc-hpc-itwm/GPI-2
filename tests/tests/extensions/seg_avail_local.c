#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_rank_t rank, nprocs;
  gaspi_number_t seg_max;
  gaspi_number_t s;
  gaspi_segment_id_t seg_avail;
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  ASSERT(gaspi_segment_max(&seg_max));
  
  assert(seg_max == GASPI_MAX_MSEGS);

  for( s = 0; s < seg_max; s++ )
    {
      ASSERT( gaspi_segment_avail_local(&seg_avail) );
      gaspi_printf("%d Creating %d of %d (%d)\n",s, seg_avail, seg_max, GASPI_MAX_MSEGS);

      ASSERT (gaspi_segment_create(seg_avail,
				   1024,
				   GASPI_GROUP_ALL,
				   GASPI_BLOCK,GASPI_MEM_UNINITIALIZED)
	      );
    }

  EXPECT_FAIL( gaspi_segment_avail_local(&seg_avail) );
  
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for( s = 0; s < seg_max; s++ )
    {
      ASSERT (gaspi_segment_delete(s));
    }

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_segment_create(0,
			       1024,
			       GASPI_GROUP_ALL,
			       GASPI_BLOCK,GASPI_MEM_UNINITIALIZED)
	  );

  ASSERT( gaspi_segment_avail_local(&seg_avail) );
  assert(seg_avail == 1);

  ASSERT (gaspi_segment_create(2,
			       1024,
			       GASPI_GROUP_ALL,
			       GASPI_BLOCK,GASPI_MEM_UNINITIALIZED)
	  );

  ASSERT( gaspi_segment_avail_local(&seg_avail) );
  assert(seg_avail == 1);

  ASSERT (gaspi_segment_create(1,
			       1024,
			       GASPI_GROUP_ALL,
			       GASPI_BLOCK,GASPI_MEM_UNINITIALIZED)
	  );

  ASSERT( gaspi_segment_avail_local(&seg_avail) );
  assert(seg_avail == 3);

  ASSERT (gaspi_segment_delete(0));
  ASSERT (gaspi_segment_delete(1));
  ASSERT (gaspi_segment_delete(2));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
