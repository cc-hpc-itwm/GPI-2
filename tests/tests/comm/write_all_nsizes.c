#include <test_utils.h>

int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;
  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _8MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_number_t queueSize, qmax;
  ASSERT (gaspi_queue_size_max(&qmax));

  const  gaspi_offset_t localOff= 0;
  const gaspi_offset_t remOff = 0;
  gaspi_size_t commSize;
  gaspi_size_t minSize;

  ASSERT(gaspi_transfer_size_min(&minSize));
  for(commSize = minSize; commSize <= _8MB; commSize*=2 )
    {
      gaspi_rank_t rankSend;
      for(rankSend = 0; rankSend < numranks; rankSend++)
	{
	  gaspi_queue_size(1, &queueSize);
	  if (queueSize > qmax - 24)
	    ASSERT (gaspi_wait(1, GASPI_BLOCK));

	  ASSERT (gaspi_write(0, localOff, rankSend, 0,  remOff,  commSize, 1, GASPI_BLOCK));
	}
    }

  ASSERT (gaspi_wait(1, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
