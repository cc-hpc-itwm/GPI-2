#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t P, myrank;

  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT(gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  const gaspi_size_t msgSize = 1;
  const int times = 1000;
  int i;

  for(i = 1; i < times; i++)
    {
      if(myrank == 0)
	{
	  gaspi_rank_t n;
	  for(n = 1; n < P; n++)
	    ASSERT(gaspi_passive_send(0, 0, n, msgSize, GASPI_BLOCK));
	}
      else
	{
	  gaspi_rank_t sender;
	  ASSERT(gaspi_passive_receive(0, 0, &sender, msgSize, GASPI_BLOCK));
	  gaspi_printf("Received msg (%lu bytes) from %d\n", msgSize, sender);
	}
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
