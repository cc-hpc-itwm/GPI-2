#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

#define MY_MAX(a,b)  (((a)<(b)) ? (b) : (a))

struct elem
  {
    double a;
    char pad[8];
  };

gaspi_return_t
my_fun (struct elem *const a,
        struct elem *const b,
        struct elem *const r,
        const gaspi_number_t num)
{
  gaspi_number_t i;

  for (i = 0; i < num; i++)
  {
    r[i].a = MY_MAX (a[i].a, b[i].a);
  }

  return GASPI_SUCCESS;
}

int
main (int argc, char *argv[])
{
  gaspi_number_t n;
  gaspi_rank_t nprocs, myrank;
  gaspi_size_t buf_size;
  gaspi_number_t elem_max;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_allreduce_buf_size (&buf_size));
  ASSERT (gaspi_allreduce_elem_max (&elem_max));

  struct elem *a = (struct elem *) calloc (elem_max, sizeof (struct elem));
  struct elem *b = (struct elem *) calloc (elem_max, sizeof (struct elem));

  if (a == NULL || b == NULL)
  {
    return EXIT_FAILURE;
  }

  for (n = 0; n < elem_max; n++)
  {
    a[n].a = b[n].a = myrank * 1.0;
  }

  if (sizeof (struct elem) * elem_max > buf_size)
  {
    elem_max = buf_size / sizeof (struct elem);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  for (n = 1; n <= elem_max; n++)
  {
    gaspi_number_t i;

    if ((n * sizeof (struct elem)) > buf_size)
    {
      EXPECT_FAIL (gaspi_allreduce_user (a, b, n, sizeof (struct elem),
                                         (gaspi_reduce_operation_t) my_fun,
                                         NULL, GASPI_GROUP_ALL, GASPI_BLOCK));
    }
    else
    {
      ASSERT (gaspi_allreduce_user (a, b, n, sizeof (struct elem),
                                    (gaspi_reduce_operation_t) my_fun, NULL,
                                    GASPI_GROUP_ALL, GASPI_BLOCK));
      for (i = 0; i < n; i++)
      {
        assert (b[i].a == nprocs - 1);
      }
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  free (a);
  free (b);

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
