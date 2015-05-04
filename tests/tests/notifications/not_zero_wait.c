#include <GASPI.h>
#include <test_utils.h>

int main(int argc, char *argv[])
{

  gaspi_rank_t r;

  gaspi_notification_id_t id;

  TSUITE_INIT(argc, argv);


  ASSERT( gaspi_proc_init(GASPI_BLOCK));
  ASSERT( gaspi_proc_rank(&r));

  ASSERT( gaspi_segment_create(0, 1, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_ALLOC_DEFAULT));


  ASSERT(gaspi_notify(0, r, r, 1, 0, GASPI_BLOCK));

  ASSERT(  gaspi_notify_waitsome(0, r, 1, &id, GASPI_BLOCK));

  // wait for zero notifications
  ASSERT( gaspi_notify_waitsome(0, 0, 0, &id, GASPI_BLOCK));

  ASSERT(  gaspi_segment_delete(0 ));

  ASSERT( gaspi_proc_term(GASPI_BLOCK) );

  return 0;

}
