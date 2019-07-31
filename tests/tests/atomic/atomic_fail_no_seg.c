#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

/* Test atomic operations without segment -> should fail */
int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t numranks, myrank, n;

  ASSERT (gaspi_proc_num (&numranks));
  ASSERT (gaspi_proc_rank (&myrank));

  gaspi_atomic_value_t val;

  for (n = 0; n < numranks; n++)
  {
    EXPECT_FAIL (gaspi_atomic_fetch_add (0, 0, n, 1, &val, GASPI_TEST));
    EXPECT_FAIL (gaspi_atomic_compare_swap
                 (0, 0, n, 1, (gaspi_atomic_value_t) 42, &val, GASPI_BLOCK));
  }

  //sync to make sure everyone did it
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
