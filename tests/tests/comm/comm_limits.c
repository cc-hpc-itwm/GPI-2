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
  
  ASSERT( gaspi_segment_create(0, max_msg_size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* communicate minimum */
  ASSERT(gaspi_read(0, 0, rank2send, 0, 0, min_msg_size, 0, GASPI_BLOCK));
  ASSERT(gaspi_wait(0, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* initialize data */
  gaspi_pointer_t _seg_ptr;
  ASSERT(gaspi_segment_ptr(0, &_seg_ptr));
  int *seg_byte_ptr = (int *) _seg_ptr;

  int byte_off;
  for(byte_off = 0; byte_off < (max_msg_size/ sizeof(int)); byte_off++)
    {
      seg_byte_ptr[byte_off] = (int)myrank;
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* communicate with max size */
  ASSERT(gaspi_read(0, 0, rank2send, 0, 0, max_msg_size, 0, GASPI_BLOCK));
  ASSERT(gaspi_wait(0, GASPI_BLOCK));
  
  for(byte_off = 0; byte_off < (max_msg_size/ sizeof(int)); byte_off++)
    {
      assert(seg_byte_ptr[byte_off] == (int) rank2send);
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* write back */
  ASSERT(gaspi_write(0, 0, rank2send, 0, 0, max_msg_size, 0, GASPI_BLOCK));
  ASSERT(gaspi_wait(0, GASPI_BLOCK));
  
  for(byte_off = 0; byte_off < (max_msg_size/ sizeof(int)); byte_off++)
    {
      assert(seg_byte_ptr[byte_off] == (int) myrank);
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
