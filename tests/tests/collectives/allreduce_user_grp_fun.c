#include <test_utils.h>
#define MY_MAX(a,b)  (((a)<(b)) ? (b) : (a))

/* Test user defined collective for a single member group and a group
 * comprised by the rest of ranks*/

gaspi_return_t
my_fun (double *const a,
        double *const b,
        double *const r,
        gaspi_state_t const state,
        const gaspi_number_t num,
        const gaspi_size_t elem_size,
        const gaspi_timeout_t tout)
{
  for (gaspi_number_t i = 0; i < num; i++)
  {
    r[i] = MY_MAX (a[i], b[i]);
  }

  return GASPI_SUCCESS;
}

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t nprocs, myrank;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  int nelems = 255;
  double *a = (double *) malloc (nelems * sizeof (double));
  double *b = (double *) malloc (nelems * sizeof (double));

  if (a == NULL || b == NULL)
  {
    return EXIT_FAILURE;
  }

  for (int n = 0; n < nelems; n++)
  {
    a[n] = b[n] = myrank * 1.0;
  }

  gaspi_group_t g;
  gaspi_number_t gsize;

  ASSERT (gaspi_group_create (&g));

  gaspi_rank_t const selected_rank = nprocs < 3 ? 0 : (nprocs / 2);

  gaspi_number_t const expected_gsize =
    myrank == selected_rank ? 1 : (nprocs-1);

  ASSERT (gaspi_group_add (g, myrank));

  if (myrank != selected_rank)
  {
    for (gaspi_rank_t i = 0; i < nprocs; i++)
    {
      if (i != selected_rank && i != myrank)
      {
        ASSERT (gaspi_group_add (g, i));
      }
    }
  }

  ASSERT (gaspi_group_size (g, &gsize));
  assert (gsize == expected_gsize);

  ASSERT (gaspi_group_commit(g, GASPI_BLOCK));

  double const expected_value =
    myrank == selected_rank ? selected_rank : (nprocs - 1);

  for (int n = 1; n <= nelems; n++)
  {
    ASSERT (gaspi_allreduce_user (a, b, n, sizeof (double),
                                  (gaspi_reduce_operation_t) my_fun, NULL,
                                  g, GASPI_BLOCK));

    for (int j = 0; j < n; j++)
    {
      assert (b[j] == expected_value);
    }
  }

  free (a);
  free (b);

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
