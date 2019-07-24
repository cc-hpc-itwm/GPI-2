#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t P, myrank;

  ASSERT (gaspi_proc_num (&P));
  ASSERT (gaspi_proc_rank (&myrank));

  gaspi_size_t passive_max_msg_size;
  gaspi_size_t passive_min_msg_size;

  ASSERT (gaspi_passive_transfer_size_max (&passive_max_msg_size));
  ASSERT (gaspi_passive_transfer_size_min (&passive_min_msg_size));

  ASSERT (gaspi_segment_create
          (0, passive_max_msg_size, GASPI_GROUP_ALL, GASPI_BLOCK,
           GASPI_MEM_INITIALIZED));

  /* initialize min data */
  gaspi_pointer_t _seg_ptr;

  ASSERT (gaspi_segment_ptr (0, &_seg_ptr));
  unsigned char *seg_byte_ptr = (unsigned char *) _seg_ptr;

  /* Minimum */
  if (myrank == 0)
  {
    memset (seg_byte_ptr, 252, passive_min_msg_size);

    gaspi_rank_t n;

    for (n = 1; n < P; n++)
    {
      ASSERT (gaspi_passive_send (0, 0, n, passive_min_msg_size, GASPI_BLOCK));
    }
  }
  else
  {
    gaspi_rank_t sender;

    ASSERT (gaspi_passive_receive
            (0, 0, &sender, passive_min_msg_size, GASPI_BLOCK));
    int elem;

    for (elem = 0; elem < passive_min_msg_size; elem++)
    {
      assert (seg_byte_ptr[elem] == 252);
    }
    memset (seg_byte_ptr, 0, passive_min_msg_size);

  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  /* Maximum */

  /* initialize max data */
  if (myrank == 0)
  {
    memset (seg_byte_ptr, 255, passive_max_msg_size);
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

    int elem;

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
