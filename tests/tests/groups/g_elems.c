#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int
main (int argc, char *argv[])
{
  gaspi_rank_t nprocs, n;
  gaspi_number_t max_groups, gsize;
  gaspi_rank_t *partners;

  TSUITE_INIT (argc, argv);

  gaspi_group_max (&max_groups);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));

  ASSERT (gaspi_group_size (GASPI_GROUP_ALL, &gsize));

  partners = malloc (gsize * sizeof (gaspi_rank_t));
  ASSERT (gaspi_group_ranks (GASPI_GROUP_ALL, partners));

  for (n = 0; n < gsize; n++)
  {
    assert (partners[n] == n);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
