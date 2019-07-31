#include <test_utils.h>

/* Test creating and committing a group without own rank */

int
main (int argc, char *argv[])
{
  gaspi_group_t g;
  gaspi_number_t ngroups, i, gsize;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t nprocs, myrank;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  //create empty group and size == 0
  ASSERT (gaspi_group_create (&g));
  ASSERT (gaspi_group_size (g, &gsize));
  assert ((gsize == 0));

  for (i = 0; i < nprocs; i++)
  {
    if (i == myrank)
    {
      continue;
    }

    ASSERT (gaspi_group_add (g, i));
  }
  ASSERT (gaspi_group_size (g, &gsize));
  assert ((gsize == (nprocs - 1)));

  EXPECT_FAIL (gaspi_group_commit (g, GASPI_BLOCK));
  ASSERT (gaspi_group_add (g, myrank));
  ASSERT (gaspi_group_commit (g, GASPI_BLOCK));
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
