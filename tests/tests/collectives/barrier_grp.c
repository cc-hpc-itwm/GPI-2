#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

#define ITERATIONS 1000

/* Test  */
int
main (int argc, char *argv[])
{
  gaspi_group_t g0, g1;
  gaspi_rank_t nprocs, myrank;
  gaspi_number_t gsize;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  gaspi_rank_t i;
  int j;

  gaspi_rank_t selected_rank = nprocs / 2;
  if (myrank == selected_rank)
  {
    // Group 0
    ASSERT (gaspi_group_create (&g0));
    ASSERT (gaspi_group_add (g0, myrank));

    ASSERT (gaspi_group_size (g0, &gsize));
    assert ((gsize == 1));

    ASSERT (gaspi_group_commit(g0, GASPI_BLOCK));

    //loop barrier
    for (j = 0; j < ITERATIONS; j++)
    {
      ASSERT (gaspi_barrier (g0, GASPI_BLOCK));

      /* with timeout */
      ASSERT (gaspi_barrier (g0, nprocs * 1000));
    }
  }
  else
  {
    // Group 1
    ASSERT (gaspi_group_create (&g1));
    for (i = 0; i < nprocs; i++)
    {
      if (i != selected_rank)
      {
        ASSERT (gaspi_group_add (g1, i));
      }
    }

    ASSERT (gaspi_group_size (g1, &gsize));
    assert ((gsize == nprocs - 1));

    ASSERT (gaspi_group_commit(g1, GASPI_BLOCK));

    //loop barrier
    for (j = 0; j < ITERATIONS; j++)
    {
      ASSERT (gaspi_barrier (g1, GASPI_BLOCK));

      /* with timeout */
      ASSERT (gaspi_barrier (g1, nprocs * 1000));
    }

  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
