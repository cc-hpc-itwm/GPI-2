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
          (0, _8MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  ASSERT (gaspi_segment_create
          (1, _8MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  //prepare memory segment
  gaspi_pointer_t _vptr, _vptr1;

  ASSERT (gaspi_segment_ptr (0, &_vptr));
  ASSERT (gaspi_segment_ptr (1, &_vptr1));

  int *mem = (int *) _vptr;
  int *mem_read = (int *) _vptr1;

  unsigned long i;
  const unsigned long maxInts = _8MB / sizeof (int);

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
  const gaspi_rank_t rank2read = (myrank + 1) % numranks;
  gaspi_segment_id_t remSegs[nListElems];
  gaspi_offset_t remOffs[nListElems];
  gaspi_size_t sizes[nListElems];

  size_t bytes = sizeof (int);
  gaspi_offset_t initLocOff = 0;
  gaspi_offset_t initRemOff = 0;

  for (gaspi_number_t n = 0; n < nListElems; n++)
  {
    sizes[n] = bytes;

    localSegs[n] = 1;
    localOffs[n] = initLocOff;
    initLocOff += bytes;

    remSegs[n] = 0;
    remOffs[n] = initRemOff;
    initRemOff += bytes;
  }

  ASSERT (gaspi_read_list_notify (nListElems,
                                  localSegs, localOffs, rank2read,
                                  remSegs, remOffs, sizes,
                                  1, myrank, 0, GASPI_BLOCK));

  gaspi_notification_id_t id;

  ASSERT (gaspi_notify_waitsome (1, myrank, 1, &id, GASPI_BLOCK));

  gaspi_notification_t notification_val;

  ASSERT (gaspi_notify_reset (1, id, &notification_val));
  assert (notification_val == 1);

  //check
  for (gaspi_number_t l = 0; l < nListElems; l++)
  {
    assert (mem_read[l] == (int) rank2read);
  }

  ASSERT (gaspi_wait (0, GASPI_BLOCK));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
