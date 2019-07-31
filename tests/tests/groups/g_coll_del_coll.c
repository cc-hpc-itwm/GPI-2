#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

/* Test 2 group collective calls with a delete in between */
/* The groups are different with one rank to avoid */

gaspi_return_t
g_create_group (gaspi_rank_t nprocs, gaspi_group_t * g, gaspi_rank_t avoid)
{
  gaspi_rank_t myrank;
  gaspi_number_t gsize;

  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_group_create (g));

  ASSERT (gaspi_group_size (*g, &gsize));
  assert ((gsize == 0));

  gaspi_rank_t i;

  for (i = 0; i < nprocs; i++)
  {
    if (i != avoid)
    {
      ASSERT (gaspi_group_add (*g, i));
    }
  }

  ASSERT (gaspi_group_size (*g, &gsize));
  assert (gsize == (nprocs - 1));

  ASSERT (gaspi_group_commit (*g, GASPI_BLOCK));

  return GASPI_SUCCESS;

}

int
main (int argc, char *argv[])
{
  gaspi_group_t g;
  gaspi_rank_t nprocs, myrank, culprit;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  if (nprocs < 3)
  {
    return EXIT_SUCCESS;
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  culprit = 1;

  if (myrank != culprit)
  {
    ASSERT (g_create_group (nprocs, &g, culprit));
    ASSERT (gaspi_barrier (g, GASPI_BLOCK));
    ASSERT (gaspi_group_delete (g));
  }

  //change culprit
  culprit = nprocs - 1;

  if (myrank != culprit)
  {
    ASSERT (g_create_group (nprocs, &g, culprit));
    ASSERT (gaspi_barrier (g, GASPI_BLOCK));
    ASSERT (gaspi_group_delete (g));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
