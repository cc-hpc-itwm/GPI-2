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

  //prepare memory segment
  gaspi_pointer_t _vptr;

  ASSERT (gaspi_segment_ptr (0, &_vptr));

  int *mem = (int *) _vptr;

  unsigned long i;
  const unsigned long maxInts = _128MB / sizeof (int);

  for (i = 0; i < maxInts; i++)
  {
    mem[i] = (int) myrank;
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  //construct list of n elems
  gaspi_number_t queue_size = 0;

  const gaspi_number_t nListElems = 255;

  gaspi_segment_id_t localSegs[nListElems];
  gaspi_offset_t localOffs[nListElems];
  const gaspi_rank_t rank2send = (myrank + 1) % numranks;
  const gaspi_rank_t rank2recv = myrank != 0 ? (myrank - 1) : (numranks - 1);
  gaspi_segment_id_t remSegs[nListElems];
  gaspi_offset_t remOffs[nListElems];
  gaspi_size_t sizes[nListElems];

  size_t bytes = sizeof (int);
  gaspi_offset_t initLocOff = 0;
  gaspi_offset_t initRemOff = (bytes * nListElems + 64);

  for (gaspi_number_t n = 0; n < nListElems; n++)
  {
    sizes[n] = bytes;

    localSegs[n] = 0;
    localOffs[n] = initLocOff;
    initLocOff += bytes;

    remSegs[n] = 0;
    remOffs[n] = initRemOff;
    initRemOff += bytes;
  }

  ASSERT (gaspi_write_list_notify (nListElems,
                                   localSegs, localOffs, rank2send,
                                   remSegs, remOffs, sizes,
                                   0, myrank, 1, 0, GASPI_BLOCK));

  ASSERT (gaspi_queue_size (0, &queue_size));

  ASSERT (gaspi_wait (0, GASPI_BLOCK));

  ASSERT (gaspi_queue_size (0, &queue_size));
  assert (queue_size == 0);

  gaspi_notification_id_t id;

  ASSERT (gaspi_notify_waitsome (0, rank2recv, 1, &id, GASPI_BLOCK));

  gaspi_notification_t notification_val;

  ASSERT (gaspi_notify_reset (0, id, &notification_val));
  assert (notification_val == 1);

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  //check
  gaspi_offset_t off2check = (bytes * nListElems + 64);
  char *chPtr = (char *) _vptr;

  mem = (int *) (chPtr + off2check);

  for (gaspi_number_t l = 0; l < nListElems; l++)
  {
    assert (mem[l] == (int) rank2recv);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
