#include <test_utils.h>

int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t P, myrank;
  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  if( P < 2 )
    {
      gaspi_printf("Must have more than 1 procs\n");
      return EXIT_SUCCESS;
    }

  ASSERT(gaspi_segment_create(0,
			      _8MB,
			      GASPI_GROUP_ALL,
			      GASPI_BLOCK,
			      GASPI_MEM_INITIALIZED));

  gaspi_pointer_t _vptr;
  ASSERT(gaspi_segment_ptr(0, &_vptr));

  gaspi_number_t qmax ;
  ASSERT (gaspi_queue_size_max(&qmax));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  unsigned long i;
  gaspi_number_t queueSize;
  int rankSend = (myrank + 1) % P;
  const unsigned long N = (1 << 10);

  for (i = 0; i < 2 * N; i ++)
    {
      gaspi_queue_size(1, &queueSize);
      if (queueSize > qmax - 24)
	{
	  gaspi_return_t ret;
	  do
	    {
	      ret = gaspi_wait(1, GASPI_TEST);
	      assert (ret != GASPI_ERROR);
	    }
	  while(ret != GASPI_SUCCESS);

	  gaspi_queue_size(1, &queueSize);
	  assert(queueSize == 0);
	}
      ASSERT (gaspi_write(0, 4, rankSend, 0, 6, 32768, 1, GASPI_TEST));
    }

  ASSERT(gaspi_wait(1, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
