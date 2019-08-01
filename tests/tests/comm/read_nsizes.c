#include <stdlib.h>
#include <stdio.h>

#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num (&numranks));
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_segment_create
          (0, _128MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  ASSERT (gaspi_segment_create
          (1, _128MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_number_t queueSize, qmax;

  ASSERT (gaspi_queue_size_max (&qmax));

  gaspi_pointer_t segPtr, segPtr1;

  ASSERT (gaspi_segment_ptr (0, &segPtr));
  int *segInt = (int *) segPtr;

  ASSERT (gaspi_segment_ptr (1, &segPtr1));
  int *segInt1 = (int *) segPtr1;

  gaspi_size_t segSize;

  ASSERT (gaspi_segment_size (0, myrank, &segSize));

  gaspi_size_t i;
  for (i = 0; i < segSize / sizeof (int); i++)
  {
    segInt[i] = myrank;
    segInt1[i] = -1;
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_size_t commSize;
  gaspi_offset_t localOff = 0;
  const gaspi_offset_t remOff = 0;

  for (commSize = sizeof (int); commSize < _8MB; commSize *= 2)
  {
    gaspi_rank_t rankSend;

    for (rankSend = 0; rankSend < numranks; rankSend++)
    {
      if (rankSend == myrank)
      {
        continue;
      }

      gaspi_queue_size (1, &queueSize);
      if (queueSize > qmax - 24)
      {
        ASSERT (gaspi_wait (1, GASPI_BLOCK));
      }

      ASSERT (gaspi_read
              (1, localOff, rankSend, 0, remOff, commSize, 1, GASPI_BLOCK));
      localOff += commSize;
    }

    ASSERT (gaspi_wait (1, GASPI_BLOCK));

    const int elems_per_rank = commSize / sizeof (int);
    int c, pos = 0;
    int *seg_read = (int *) segPtr1;

    for (rankSend = 0; rankSend < numranks; rankSend++)
    {
      if (rankSend == myrank)
        continue;

      for (c = 0; c < elems_per_rank; c++)
      {
        assert (seg_read[pos] == rankSend);
        pos++;
      }
    }

    localOff = 0;
  }

  ASSERT (gaspi_wait (1, GASPI_BLOCK));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
