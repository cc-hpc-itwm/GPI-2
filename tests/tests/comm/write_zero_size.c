#include <test_utils.h>

/* Test for write of zero size messages */

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nprocs, i;
  const gaspi_segment_id_t seg_id = 0;
  gaspi_offset_t offset;

  gaspi_number_t queue_size;
  gaspi_number_t queue_max;

  ASSERT (gaspi_queue_size_max (&queue_max));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  ASSERT (gaspi_segment_create
          (seg_id, nprocs * sizeof (int), GASPI_GROUP_ALL, GASPI_BLOCK,
           GASPI_MEM_UNINITIALIZED));

  offset = rank * sizeof (int);

  gaspi_pointer_t _vptr;

  ASSERT (gaspi_segment_ptr (0, &_vptr));

  int *mem = (int *) _vptr;

  for (i = 0; i < nprocs; i++)
  {
    mem[i] = (int) rank;
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  for (i = 0; i < nprocs; i++)
  {
    ASSERT (gaspi_queue_size (0, &queue_size));
    if (queue_size > queue_max - 1)
    {
      ASSERT (gaspi_wait (0, GASPI_BLOCK));
    }

    ASSERT (gaspi_write (seg_id, offset, i,
                         seg_id, offset, 0,
                         0, GASPI_BLOCK));
  }

  ASSERT (gaspi_wait (0, GASPI_BLOCK));


  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  for (i = 0; i < nprocs; i++)
  {
    assert (mem[i] == (int) rank);
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
