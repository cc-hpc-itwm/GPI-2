#include <test_utils.h>

int
main (int argc, char *argv[])
{
  gaspi_group_t g;
  gaspi_rank_t nprocs, myrank;
  gaspi_number_t gsize;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_group_create (&g));

  gaspi_rank_t i;

  for (i = 0; i < nprocs; i++)
  {
    ASSERT (gaspi_group_add (g, i));
  }

  ASSERT (gaspi_group_size (g, &gsize));
  assert ((gsize == nprocs));

  if (myrank > 0)
  {
    sleep (5);                  //simulate delay
  }

  gaspi_return_t ret;

  do
  {
    ret = gaspi_group_commit (g, 1000);

  }
  while (ret == GASPI_TIMEOUT || ret == GASPI_ERROR);

  assert ((ret != GASPI_ERROR));

  ASSERT (gaspi_barrier (g, 10000));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
