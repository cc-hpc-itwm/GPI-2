#include <test_utils.h>

/* Test timeout in passive_receive without sender */
int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t P, myrank;

  ASSERT (gaspi_proc_num (&P));
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_segment_create
          (0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  const gaspi_size_t msgSize = 4;

  if (myrank != 0)
  {
    gaspi_rank_t sender;

    EXPECT_TIMEOUT (gaspi_passive_receive (0, 0, &sender, msgSize, 2000));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
