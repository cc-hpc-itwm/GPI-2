#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

/* in a loop: */
/* create a segment, use it, delete it (two times) */
/* repeat ntimes */

int main(int argc, char *argv[])
{
  int ntimes = 10;

  gaspi_rank_t rank, nprocs;
  gaspi_notification_id_t id;
  gaspi_notification_t val;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));
  const gaspi_rank_t right = (rank + nprocs + 1) % nprocs;

  do
    {
      ASSERT (gaspi_segment_create(0, 1024,
				   GASPI_GROUP_ALL,
				   GASPI_BLOCK,
				   GASPI_MEM_UNINITIALIZED));

      ASSERT( gaspi_write_notify(0, 0, right,
				 0, 0, 8,
				 0, 1,
				 0, GASPI_BLOCK) );
      ASSERT( gaspi_wait( 0, GASPI_BLOCK) );

      ASSERT(gaspi_notify_waitsome(0, 0, 1, &id, GASPI_BLOCK));
      ASSERT( gaspi_notify_reset(0, id, &val));

      ASSERT (gaspi_segment_delete(0));

      ASSERT (gaspi_segment_create(0, 2048,
				   GASPI_GROUP_ALL,
				   GASPI_BLOCK,
				   GASPI_MEM_UNINITIALIZED));


      ASSERT( gaspi_write_notify(0, 0, right,
				 0, 0, 8,
				 0, 1,
				 0, GASPI_BLOCK) );
      ASSERT( gaspi_wait( 0, GASPI_BLOCK) );

      ASSERT(gaspi_notify_waitsome(0, 0, 1, &id, GASPI_BLOCK));
      ASSERT( gaspi_notify_reset(0, id, &val));

      ASSERT (gaspi_segment_delete(0));
      ntimes--;
    }
  while(ntimes > 0);

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
