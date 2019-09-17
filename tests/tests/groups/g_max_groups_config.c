#include <test_utils.h>

//Test sets the max number of groups in config, creates and commits
//them (with all ranks) and then deletes them.

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  gaspi_rank_t nprocs, n;

  gaspi_number_t const user_max_groups = 255;

  gaspi_config_t config;
  ASSERT (gaspi_config_get (&config));

  config.group_max = user_max_groups;
  ASSERT (gaspi_config_set (config));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));

  gaspi_number_t max_groups, i;

  gaspi_group_max (&max_groups);
  assert (max_groups == user_max_groups);

  gaspi_group_t* gs = calloc (user_max_groups, sizeof (gaspi_group_t));
  if (gs == NULL)
  {
    return EXIT_FAILURE;
  }

  for (i = 1; i < max_groups; i++)
  {
    ASSERT (gaspi_group_create (&(gs[i])));
  }

  for (i = 1; i < max_groups; i++)
  {
    for (n = 0; n < nprocs; n++)
    {
      ASSERT (gaspi_group_add (gs[i], n));
    }
  }

  for (i = 1; i < max_groups; i++)
  {
    ASSERT (gaspi_group_commit (gs[i], GASPI_BLOCK));
  }

  gaspi_number_t groups_created;
  ASSERT (gaspi_group_num (&groups_created));
  assert (groups_created == user_max_groups);

  for (i = 0; i < max_groups; i++)
  {
    ASSERT (gaspi_barrier (gs[i], GASPI_BLOCK));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  for (i = 1; i < groups_created; i++)
  {
    ASSERT (gaspi_group_delete (gs[i]));
  }

  EXPECT_FAIL_WITH (gaspi_group_delete (GASPI_GROUP_ALL),
                    GASPI_ERR_INV_GROUP);

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
