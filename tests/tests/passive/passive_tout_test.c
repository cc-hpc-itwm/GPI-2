#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

/* Test that sends passive messages without blocking (GASPI_TEST) */
int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t P, myrank;

  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  if( P < 2 )
    {
      return EXIT_SUCCESS;
    }

  ASSERT (gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  const gaspi_size_t msgSize = 4;
  if(P > 1)
    {
      
      if(myrank == 1)
	{
	  gaspi_rank_t n;
	  for(n = 0; n < P; n++)
	    {
	      if(n == myrank)
		continue;
	  
	      gaspi_return_t ret = GASPI_ERROR;
	      do
		{
		  ret = gaspi_passive_send(0, 0, n, msgSize, GASPI_TEST);
		}
	      while(ret != GASPI_SUCCESS);
	    }
	}
      else
	{
	  gaspi_rank_t sender;
	  ASSERT(gaspi_passive_receive(0, 0, &sender, msgSize, GASPI_BLOCK));
	  assert(sender == 1);
	}
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
