#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_group_t gs[32];

  gaspi_rank_t nprocs, n;
  gaspi_number_t max_groups,i;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));

  gaspi_group_max(&max_groups);

  for(i = 0; i < max_groups - 1; i++)
    {
      ASSERT(gaspi_group_create(&(gs[i])));
    }

  for(i = 0; i < max_groups - 1; i++)
    for(n = 0; n < nprocs; n++)
      ASSERT(gaspi_group_add(gs[i], n));

  for(i = 0; i < max_groups - 1; i++)
    ASSERT(gaspi_group_commit(gs[i], GASPI_BLOCK));

  for(i = 0; i < max_groups - 1; i++)
    ASSERT (gaspi_barrier(gs[i], GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
