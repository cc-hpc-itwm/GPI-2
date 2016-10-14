#include <test_utils.h>

/* Test limits of communication (min and max) */
int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t P, myrank, rank2send;
  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  rank2send = (myrank + 1) % P;
  assert(rank2send < P);

  gaspi_size_t max_msg_size;
  ASSERT( gaspi_transfer_size_max (&max_msg_size));

  gaspi_size_t min_msg_size;
  ASSERT( gaspi_transfer_size_min (&min_msg_size));

  ASSERT( gaspi_segment_create(0, max_msg_size * 2, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* communicate minimum */
  ASSERT(gaspi_read(0, 0, rank2send, 0, 0, min_msg_size, 0, GASPI_BLOCK));
  ASSERT(gaspi_wait(0, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* initialize data */
  gaspi_pointer_t _seg_ptr;
  ASSERT(gaspi_segment_ptr(0, &_seg_ptr));
  unsigned char *seg_byte_ptr = (unsigned char *) _seg_ptr;

  memset(seg_byte_ptr, 255, max_msg_size);

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* communicate with max size */
  ASSERT(gaspi_read(0, max_msg_size, rank2send, 0, 0, max_msg_size, 0, GASPI_BLOCK));
  ASSERT(gaspi_wait(0, GASPI_BLOCK));

  int elem;
  for(elem = 0; elem < max_msg_size * 2; elem++)
    {
      assert(seg_byte_ptr[elem] == 255 );
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* reset data */
  memset(seg_byte_ptr, 254, max_msg_size);

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* write back */
  ASSERT(gaspi_write_notify(0, 0, rank2send, 0, max_msg_size, max_msg_size, 0, 1, 0, GASPI_BLOCK));
  ASSERT(gaspi_wait(0, GASPI_BLOCK));

  gaspi_notification_id_t id;
  ASSERT(gaspi_notify_waitsome(0, 0, 1, &id, GASPI_BLOCK));

  gaspi_notification_t val;
  ASSERT(gaspi_notify_reset(0, id, &val));
  assert(val == 1);

  for(elem = 0; elem < max_msg_size * 2; elem++)
    {
      assert(seg_byte_ptr[elem] == 254);
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
