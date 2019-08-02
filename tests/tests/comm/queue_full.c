#include <stdlib.h>
#include <stdio.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  int rankSend = (myrank + 1) % numranks;

  ASSERT(gaspi_segment_create(0, _1MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_size_t segSize;
  ASSERT( gaspi_segment_size(0, myrank, &segSize));

  unsigned char * pGlbMem;

  gaspi_pointer_t _vptr;
  ASSERT(gaspi_segment_ptr(0, &_vptr));

  pGlbMem = ( unsigned char *) _vptr;

  gaspi_number_t queueSize, qmax ;
  ASSERT (gaspi_queue_size_max(&qmax));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  const unsigned long localOff = 0;
  const unsigned long remOff = 0;

  /* write_notify */
  do
    {
      ASSERT(gaspi_write_notify(0, localOff, rankSend,
                                0, remOff, 1,
                                (gaspi_notification_id_t) myrank, 1,
                                1, GASPI_BLOCK));
      gaspi_queue_size(1, &queueSize);
    }
  while(queueSize < qmax);

  EXPECT_FAIL_WITH(gaspi_write_notify(0, localOff, rankSend,
                                      0, remOff, 1,
                                      (gaspi_notification_id_t) myrank, 1,
                                      1, GASPI_BLOCK),
                   GASPI_QUEUE_FULL);

  ASSERT (gaspi_wait(1, GASPI_BLOCK));

  /* write */
  do
    {
      ASSERT(gaspi_write(0, localOff, rankSend,
                         0, remOff, 1,
                         1, GASPI_BLOCK));

      gaspi_queue_size(1, &queueSize);
    }
  while(queueSize < qmax);

  EXPECT_FAIL_WITH(gaspi_write(0, localOff, rankSend,
                               0, remOff, 1,
                               1, GASPI_BLOCK),
                   GASPI_QUEUE_FULL);

  ASSERT (gaspi_wait(1, GASPI_BLOCK));

  ASSERT(gaspi_write(0, localOff, rankSend,
                     0, remOff, 1,
                     1, GASPI_BLOCK));

  /* write + write_notify */
  do
    {
      ASSERT(gaspi_write(0, localOff, rankSend,
                         0, remOff, 1,
                         1, GASPI_BLOCK));

      gaspi_queue_size(1, &queueSize);
    }
  while(queueSize < qmax - 1);

  EXPECT_FAIL_WITH(gaspi_write_notify(0, localOff, rankSend,
                                      0, remOff, 1,
                                      (gaspi_notification_id_t) myrank, 1,
                                      1, GASPI_BLOCK),
                   GASPI_QUEUE_FULL);

  ASSERT (gaspi_wait(1, GASPI_BLOCK));

  ASSERT(gaspi_write_notify(0, localOff, rankSend,
                            0, remOff, 1,
                            (gaspi_notification_id_t) myrank, 1,
                            1, GASPI_BLOCK));

  /* read */
  do
    {
      ASSERT(gaspi_read(0, localOff, rankSend,
                        0, remOff, 1,
                        1, GASPI_BLOCK));

      gaspi_queue_size(1, &queueSize);
    }
  while(queueSize < qmax);

  EXPECT_FAIL_WITH(gaspi_read(0, localOff, rankSend,
                              0, remOff, 1,
                              1, GASPI_BLOCK),
                   GASPI_QUEUE_FULL);

  ASSERT (gaspi_wait(1, GASPI_BLOCK));

  ASSERT(gaspi_read(0, localOff, rankSend,
                    0, remOff, 1,
                    1, GASPI_BLOCK));

  /* write_list_notify */
  {
    const gaspi_number_t nListElems = 255;
    gaspi_number_t n;

    gaspi_segment_id_t localSegs[nListElems];
    gaspi_offset_t localOffs[nListElems];
    const gaspi_rank_t rank2send = (myrank + 1) % numranks;
    gaspi_segment_id_t remSegs[nListElems];
    gaspi_offset_t remOffs[nListElems];
    gaspi_size_t sizes[nListElems];

    const unsigned int bytes = sizeof(int);
    gaspi_offset_t initLocOff = 0;
    gaspi_offset_t initRemOff = (bytes * nListElems + 64);

    for(n = 0; n < nListElems; n++)
    {
      sizes[n] = bytes;

      localSegs[n] = 0;
      localOffs[n] = initLocOff;
      initLocOff += bytes;

      remSegs[n] = 0;
      remOffs[n] = initRemOff;
      initRemOff += bytes;
    }

    do
      {
        ASSERT( gaspi_write_list_notify( nListElems,
                                         localSegs, localOffs, rank2send,
                                         remSegs, remOffs, sizes,
                                         0, myrank, 1,
                                         0, GASPI_BLOCK));

        gaspi_queue_size(0, &queueSize);
      }
    while(queueSize < qmax);

    EXPECT_FAIL_WITH( gaspi_write_list_notify( nListElems,
                                               localSegs, localOffs, rank2send,
                                               remSegs, remOffs, sizes,
                                               0, myrank, 1,
                                               0, GASPI_BLOCK),
                      GASPI_QUEUE_FULL);
    ASSERT (gaspi_wait(0, GASPI_BLOCK));
    ASSERT( gaspi_write_list_notify( nListElems,
                                     localSegs, localOffs, rank2send,
                                     remSegs, remOffs, sizes,
                                     0, myrank, 1,
                                     0, GASPI_BLOCK));

  }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
