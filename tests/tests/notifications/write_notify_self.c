#include <test_utils.h>
#include <stdio.h>

static gaspi_return_t
write_and_notify_self (gaspi_segment_id_t seg_id,
                       gaspi_offset_t offset,
                       gaspi_rank_t target,
                       gaspi_size_t size)
{
  gaspi_number_t queue_size;
  gaspi_number_t queue_max;

  ASSERT (gaspi_queue_size_max (&queue_max));
  ASSERT (gaspi_queue_size (0, &queue_size));

  gaspi_rank_t myrank;
  ASSERT (gaspi_proc_rank (&myrank));

  if (queue_size > queue_max - 1)
  {
    ASSERT (gaspi_wait (0, GASPI_BLOCK));
  }

  ASSERT (gaspi_write_notify_self (seg_id, offset, target,
                                   seg_id, offset, size,
                                   (gaspi_notification_id_t) target,
                                   0, GASPI_BLOCK));

  gaspi_notification_id_t id;
  ASSERT (gaspi_notify_waitsome
          (seg_id, target, 1, &id, GASPI_BLOCK));
  assert (id == target);

  return GASPI_SUCCESS;
}
int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nprocs;
  const gaspi_segment_id_t seg_id = 0;
  gaspi_offset_t offset;


  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  if (nprocs < 2)
  {
    return EXIT_SUCCESS;
  }

  gaspi_size_t size_to_test = _1GB;

  ASSERT (gaspi_segment_create
          (seg_id, size_to_test, GASPI_GROUP_ALL, GASPI_BLOCK,
           GASPI_MEM_INITIALIZED));

  offset = rank * sizeof (int);

  gaspi_pointer_t _vptr;

  ASSERT (gaspi_segment_ptr (0, &_vptr));

  char* mem = (char*) _vptr;

  /* Rank 0 writes the full size to all other ranks */
  if (rank == 0)
  {
    for (gaspi_rank_t i = nprocs - 1; i > 0; i--)
    {
      memset (mem, i, size_to_test);

      ASSERT (write_and_notify_self (seg_id, offset, i, size_to_test));
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (rank != 0)
  {
    for (gaspi_size_t i = 0; i < size_to_test; i++)
    {
      assert (mem[i] == rank);
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  /* All ranks write their part of full size to all other ranks */
  memset (mem, rank, size_to_test);
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_size_t size_per_rank = size_to_test / nprocs;
  offset = rank * size_per_rank;

  for (gaspi_rank_t i = 0; i < nprocs; i++)
  {
    ASSERT (write_and_notify_self (seg_id, offset, i, size_per_rank));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  for (gaspi_rank_t r = 0; r < nprocs; r++)
  {
    char* mem_part = mem + (r * size_per_rank);
    for (gaspi_size_t i = 0; i < size_per_rank; i++)
    {
      assert (mem_part[i] == r);
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
