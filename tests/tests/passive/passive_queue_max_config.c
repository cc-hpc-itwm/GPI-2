#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  gaspi_config_t config;
  ASSERT (gaspi_config_get (&config));

  config.passive_queue_size_max = 1024;
  ASSERT (gaspi_config_set (config));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t P, myrank, rank2send;

  ASSERT (gaspi_proc_num (&P));
  ASSERT (gaspi_proc_rank (&myrank));

  rank2send = (myrank + 1) % P;
  assert (rank2send < P);

  gaspi_size_t const segment_size = 1024;
  ASSERT (gaspi_segment_create
          (0, segment_size, GASPI_GROUP_ALL, GASPI_BLOCK,
           GASPI_MEM_INITIALIZED));

  gaspi_pointer_t _seg_ptr;

  ASSERT (gaspi_segment_ptr (0, &_seg_ptr));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  unsigned char *seg_byte_ptr = (unsigned char *) _seg_ptr;

  if (myrank == 0)
  {
    memset (seg_byte_ptr, 255, segment_size);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (myrank == 0)
  {
    for (gaspi_rank_t n = 1; n < P; n++)
    {
      ASSERT (gaspi_passive_send (0, 0, n, segment_size, GASPI_BLOCK));
    }
  }
  else
  {
    gaspi_rank_t sender;

    ASSERT (gaspi_passive_receive
            (0, 0, &sender, segment_size, GASPI_BLOCK));

    for (gaspi_size_t elem = 0; elem < segment_size; elem++)
    {
      assert (seg_byte_ptr[elem] == 255);
    }
    memset (seg_byte_ptr, 0, segment_size);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));
}
