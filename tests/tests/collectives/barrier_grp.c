#include <test_utils.h>
#define ITERATIONS 1000

/* Test barrier for a single member group and a group comprised by the
 * rest of ranks*/

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t nprocs, myrank;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  gaspi_group_t g;
  gaspi_number_t gsize;

  ASSERT (gaspi_group_create (&g));

  gaspi_rank_t selected_rank = nprocs / 2;

  if (myrank == selected_rank)
  {
    ASSERT (gaspi_group_add (g, myrank));

    ASSERT (gaspi_group_size (g, &gsize));
    assert ((gsize == 1));
  }
  else
  {
    for (gaspi_rank_t i = 0; i < nprocs; i++)
    {
      if (i != selected_rank)
      {
        ASSERT (gaspi_group_add (g, i));
      }
    }

    ASSERT (gaspi_group_size (g, &gsize));
    assert ((gsize == nprocs - 1));
  }

  ASSERT (gaspi_group_commit(g, GASPI_BLOCK));

  // loop barrier
  for (int j = 0; j < ITERATIONS; j++)
  {
    /* with blocking */
    ASSERT (gaspi_barrier (g, GASPI_BLOCK));
    /* with timeout */
    ASSERT (gaspi_barrier (g, nprocs * 1000));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
