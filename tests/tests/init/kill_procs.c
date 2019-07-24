#include <test_utils.h>

int
main (int argc, char *argv[])
{
  gaspi_rank_t rank, nc, i;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_rank (&rank));

  ASSERT (gaspi_proc_num (&nc));

  if (rank == 0)
  {
    for (i = 1; i < nc; i++)
    {
      ASSERT (gaspi_proc_kill (i, GASPI_BLOCK));
    }
    EXPECT_FAIL (gaspi_proc_kill (rank, GASPI_BLOCK));
  }
  else
  {
    //hang in there
    ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
