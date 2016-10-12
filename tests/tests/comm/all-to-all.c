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

  ASSERT (gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  const gaspi_offset_t localOff = 0;
  const gaspi_offset_t remOff   = 0;
  const gaspi_offset_t size = 1;
  gaspi_number_t queueSize, qmax;
  const gaspi_queue_id_t q = 0;

  ASSERT (gaspi_queue_size_max(&qmax));

  gaspi_rank_t rankSend;
  for(rankSend = 0; rankSend < numranks; rankSend++)
    {
      gaspi_queue_size(q, &queueSize);
      if( queueSize > qmax - 24 )
	{
	  ASSERT (gaspi_wait(q, GASPI_BLOCK));
	}

      ASSERT (gaspi_write(0, localOff, rankSend, 0,  remOff,  size, q, GASPI_BLOCK));
    }

  ASSERT (gaspi_wait(q, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
