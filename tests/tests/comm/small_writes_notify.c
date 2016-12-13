#include <stdlib.h>
#include <stdio.h>

#include <test_utils.h>

/*
  Test to replicate issue on Cray.

  Write large buffer in chunks using many small msgs (< 8K) with a
  final notify for data arrival.

 */
int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  int rankSend = (myrank + 1) % numranks;

  ASSERT(gaspi_segment_create(0, _1GB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_size_t segSize;
  ASSERT( gaspi_segment_size(0, myrank, &segSize));

  const gaspi_size_t half_segSize = (segSize / 2);

  int * pGlbMem;

  gaspi_pointer_t _vptr;
  ASSERT(gaspi_segment_ptr(0, &_vptr));

  pGlbMem = ( int *) _vptr;

  int i;
  for(i = 0; i < segSize / sizeof(int); i++)
    {
      pGlbMem[i] = myrank;
    }

  gaspi_number_t qmax ;
  ASSERT (gaspi_queue_size_max(&qmax));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  unsigned long localOff = 0;
  unsigned long remOff = half_segSize;
  const gaspi_size_t chunkSize = 4096;
  gaspi_size_t data_transfered = 0;
  gaspi_number_t queueSize;

  do
    {
      ASSERT(gaspi_queue_size(0, &queueSize));
      if( queueSize  >= qmax )
	{
	  ASSERT(gaspi_wait(0, GASPI_BLOCK));
	}

      ASSERT( gaspi_write(0, localOff, rankSend,
			  0, remOff, chunkSize,
			  0, GASPI_BLOCK));
      data_transfered += chunkSize;
      remOff += chunkSize;
      localOff += chunkSize;
    }
  while(data_transfered < half_segSize);

  /* Wait long enough to make sure data is transfered. Uncomment for
     debugging. */
  /* sleep(10); */

  ASSERT(gaspi_queue_size(0, &queueSize));
  if( queueSize  >= qmax )
    {
      ASSERT(gaspi_wait(0, GASPI_BLOCK));
    }

  ASSERT(gaspi_notify(0, rankSend, 0, 1, 0, GASPI_BLOCK));

  gaspi_notification_id_t got;
  ASSERT(gaspi_notify_waitsome(0, 0, 1, &got, GASPI_BLOCK));
  assert(got == 0);

  gaspi_notification_t got_val;
  ASSERT(gaspi_notify_reset(0, got, &got_val));
  assert(got_val == 1);

  for(i = (half_segSize / sizeof(int)); i < (segSize / sizeof(int)); i++)
    {
      assert(pGlbMem[i] == (myrank + numranks - 1) % numranks);
    }

  ASSERT(gaspi_wait(0, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
