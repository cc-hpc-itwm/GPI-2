#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  gaspi_config_t config;
  ASSERT (gaspi_config_get (&config));

  gaspi_size_t user_defined_transfer_max = 1024;
  config.passive_transfer_size_max = user_defined_transfer_max;
  ASSERT (gaspi_config_set (config));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t P, myrank;

  ASSERT (gaspi_proc_num (&P));
  ASSERT (gaspi_proc_rank (&myrank));

  gaspi_size_t passive_max_msg_size;

  ASSERT (gaspi_passive_transfer_size_max (&passive_max_msg_size));

  ASSERT (gaspi_segment_create
          (0, passive_max_msg_size, GASPI_GROUP_ALL, GASPI_BLOCK,
           GASPI_MEM_INITIALIZED));

  /* initialize min data */
  gaspi_pointer_t _seg_ptr;

  ASSERT (gaspi_segment_ptr (0, &_seg_ptr));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  /* initialize max data */
  unsigned char *seg_byte_ptr = (unsigned char *) _seg_ptr;

  if (myrank == 0)
  {
    memset (seg_byte_ptr, 255, passive_max_msg_size);
    EXPECT_FAIL (gaspi_passive_send (0, 0, 1,
                                     passive_max_msg_size + 1, GASPI_BLOCK));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (myrank == 0)
  {
    gaspi_rank_t n;

    for (n = 1; n < P; n++)
    {
      ASSERT (gaspi_passive_send (0, 0, n, passive_max_msg_size, GASPI_BLOCK));
    }
  }
  else
  {
    gaspi_rank_t sender;

    ASSERT (gaspi_passive_receive
            (0, 0, &sender, passive_max_msg_size, GASPI_BLOCK));

    gaspi_size_t elem;

    for (elem = 0; elem < passive_max_msg_size; elem++)
    {
      assert (seg_byte_ptr[elem] == 255);
    }
    memset (seg_byte_ptr, 0, passive_max_msg_size);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
