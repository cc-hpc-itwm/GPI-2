#include <test_utils.h>

/* Test allocation of a segment of different size in rank 0 and other
   ranks and register it with all other ranks. Then delete it. */

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_rank_t rank, nprocs;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  if (rank == 0)
  {
    ASSERT (gaspi_segment_alloc (0, 1024, GASPI_MEM_INITIALIZED));
  }
  else
  {
    ASSERT (gaspi_segment_alloc (0, 2048, GASPI_MEM_INITIALIZED));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_rank_t i;

  for (i = 0; i < nprocs; i++)
  {
    if (i == rank)
    {
      continue;
    }

    ASSERT (gaspi_segment_register (0, i, GASPI_BLOCK));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_segment_delete (0));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
