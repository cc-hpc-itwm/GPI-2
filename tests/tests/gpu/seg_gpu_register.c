#include <stdio.h>
#include <stdlib.h>
#include <GASPI_GPU.h>
#include <test_utils.h>
#include <cuda.h> 
//alloc max number of segs, fail after that
//and them register them all with all nodes
//then delete them
int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  //need the barrier to make sn is up
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_rank_t rank, nprocs, i;
  gaspi_number_t seg_max;

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  ASSERT(gaspi_init_GPUs());
  cudaSetDevice(0);
  seg_max = 1;
  
 // return 0;
  ASSERT (gaspi_segment_create(0, 1024,GASPI_GROUP_ALL,GASPI_BLOCK, GASPI_MEM_INITIALIZED|GASPI_MEM_GPU ));


  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  for (i = 0; i < nprocs; i++)
    {

      
      if(i == rank)
	continue;

      ASSERT( gaspi_segment_register(0, i, GASPI_BLOCK));
      //      sleep(1);
    }

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_segment_delete(0));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
