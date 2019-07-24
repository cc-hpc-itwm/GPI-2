#include <test_utils.h>

/* Test allocates max number of segments, registers them with all other nodes and
   then deletes them */
int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nprocs, i;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  gaspi_number_t seg_max;

  ASSERT (gaspi_segment_max (&seg_max));

  assert (seg_max == 32);

  gaspi_segment_id_t s;

  for (s = 0; s < seg_max; s++)
  {
    ASSERT (gaspi_segment_alloc (s, 1024, GASPI_MEM_INITIALIZED));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  EXPECT_FAIL (gaspi_segment_alloc (s, 1024, GASPI_MEM_INITIALIZED));

  for (i = 0; i < nprocs; i++)
  {
    if (i == rank)
    {
      continue;
    }

    for (s = 0; s < seg_max; s++)
    {
      ASSERT (gaspi_segment_register (s, i, GASPI_BLOCK));
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  for (s = 0; s < seg_max; s++)
  {
    ASSERT (gaspi_segment_delete (s));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
