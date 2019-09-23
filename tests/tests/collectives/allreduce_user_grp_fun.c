#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

#define MY_MAX(a,b)  (((a)<(b)) ? (b) : (a))

/* Checks that a user defined reduction works in a single member group
 * and a group comprised by the rest of the ranks. */

gaspi_return_t
my_fun (double *const a,
        double *const b,
        double *const r,
        gaspi_state_t const state,
        const gaspi_number_t num,
        const gaspi_size_t elem_size,
        const gaspi_timeout_t tout)
{
  gaspi_number_t i;

  for (i = 0; i < num; i++)
  {
    r[i] = MY_MAX (a[i], b[i]);
  }

  return GASPI_SUCCESS;
}

int
main (int argc, char *argv[])
{
  gaspi_rank_t nprocs, myrank;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  int nelems = 255;
  double *a = (double *) malloc (nelems * sizeof (double));
  double *b = (double *) malloc (nelems * sizeof (double));

  if (a == NULL || b == NULL)
  {
    return EXIT_FAILURE;
  }

  int n;
  for (n = 0; n < nelems; n++)
  {
    a[n] = b[n] = myrank * 1.0;
  }

  int i;
  gaspi_group_t g0, g1;
  gaspi_number_t gsize;

  if (myrank == 0)
  {
    // Group 0
    ASSERT (gaspi_group_create (&g0));
    ASSERT (gaspi_group_add (g0, myrank));

    ASSERT (gaspi_group_size (g0, &gsize));
    assert ((gsize == 1));

    ASSERT (gaspi_group_commit(g0, GASPI_BLOCK));

    for (n = 1; n <= nelems; n++)
    {
      ASSERT (gaspi_allreduce_user (a, b, n, sizeof (double),
                                    (gaspi_reduce_operation_t) my_fun, NULL,
                                    g0, GASPI_BLOCK));

      for (i = 0; i < n; i++)
      {
        assert (b[i] == 0.0);
      }
    }
  }
  else
  {
    // Group 1
    ASSERT (gaspi_group_create (&g1));
    for (i = 1; i < nprocs; i++)
    {
      ASSERT (gaspi_group_add (g1, i));
    }

    ASSERT (gaspi_group_size (g1, &gsize));
    assert ((gsize == nprocs - 1));

    ASSERT (gaspi_group_commit(g1, GASPI_BLOCK));

    for (n = 1; n <= nelems; n++)
    {
      ASSERT (gaspi_allreduce_user (a, b, n, sizeof (double),
                                    (gaspi_reduce_operation_t) my_fun, NULL,
                                    g1, GASPI_BLOCK));
      for (i = 0; i < n; i++)
      {
        assert (b[i] == (double) (nprocs - 1));
      }
    }
  }

  free (a);
  free (b);

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
