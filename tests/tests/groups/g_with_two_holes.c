#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <test_utils.h>

/* Test assigment of several segments in a groups skipping the
   addition of two no adjacent ranks */
int
main (int argc, char *argv[])
{
  gaspi_number_t i, gsize,  nsegm = 10;
  gaspi_group_t g;
  gaspi_rank_t nprocs, myrank, culprit1 = 1, culprit2 = 3;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  if (nprocs < 3)
  {
    return EXIT_SUCCESS;
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (myrank != culprit1 && myrank != culprit2)
  {

    ASSERT (gaspi_group_create (&g));
    for (i = 0; i < nprocs; i++)
    {
      if (i != culprit1 && i != culprit2)
      {
        ASSERT (gaspi_group_add (g, i));
      }
    }

    ASSERT (gaspi_group_size (g, &gsize));
    assert (gsize == (nprocs - 2));

    ASSERT (gaspi_group_commit (g, GASPI_BLOCK));

    ASSERT (gaspi_barrier (g, GASPI_BLOCK));

    for (i = 0; i < nsegm; ++i)
    {
      ASSERT (gaspi_segment_create
              (i, 1024 * 1024, g, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
    }
  }

  gaspi_number_t segment_num;
  ASSERT (gaspi_segment_num (&segment_num));

  gaspi_segment_id_t *segment_id_list;
  segment_id_list =
    (gaspi_segment_id_t *) malloc (segment_num * sizeof (gaspi_segment_id_t));
  ASSERT (gaspi_segment_list(segment_num, segment_id_list));
  assert (segment_id_list != NULL);

  for (i = 0; i < segment_num; ++i)
  {
    assert (segment_id_list[i] == i);
  }

  if (myrank != culprit1 && myrank != culprit2)
  {
    for (i = 0; i < nsegm; ++i)
    {
      ASSERT (gaspi_segment_delete(i));
    }
    ASSERT (gaspi_group_delete (g));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
