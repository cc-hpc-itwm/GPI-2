#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_notification_id_t notif = 0;
  gaspi_number_t notif_num, n;
  gaspi_rank_t rank, nprocs, i;
  const  gaspi_segment_id_t seg_id = 0;

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  ASSERT (gaspi_segment_create(seg_id, 1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT( gaspi_notification_num(&notif_num));

  ASSERT (gaspi_notify( seg_id, (rank + 1) % nprocs, (notif_num - 1), 1, 0, GASPI_BLOCK));

  gaspi_notification_id_t id;
  EXPECT_FAIL (gaspi_notify_waitsome(seg_id, notif_num - 1, 2, &id, GASPI_BLOCK));
  ASSERT (gaspi_notify_waitsome(seg_id, notif_num - 1, 1, &id, GASPI_BLOCK));

  gaspi_notification_t notification_val;
  ASSERT( gaspi_notify_reset(seg_id, id, &notification_val));
  assert( notification_val == 1 );

  ASSERT(gaspi_wait(0, GASPI_BLOCK));
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
