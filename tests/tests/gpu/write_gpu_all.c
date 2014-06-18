#include <stdio.h>
#include <stdlib.h>
#include <GASPI_GPU.h>
#include <test_utils.h>
#include <cuda.h>

#define _128MB 134217728
//create GPU segemnts on all nodes
//and then tranfer data between all segments
int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);


  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  //need the barrier to make sn is up
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_rank_t rank, nprocs, i,j;
  gaspi_number_t seg_max;

  gaspi_gpu_t gpus[8]; 
  gaspi_gpu_num nGPUs;



  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));
  ASSERT(gaspi_init_GPUs());
  seg_max = 1;
  ASSERT (gaspi_number_of_GPUs(&nGPUs));
  ASSERT (gaspi_GPU_ids(gpus));

  for (i =0; i<nGPUs; i++){
    cudaSetDevice(gpus[i]);
    ASSERT (gaspi_segment_create(i, _128MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED|GASPI_MEM_GPU));
  }

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_number_t queueSize;

  int rankSend = (rank + 1) % nprocs;

  for (j =0; j<nGPUs; j++){

    //sleep(1);
    if (gaspi_gpu_write(j, //seg
          1024, //local
          rankSend, //rank
          j, //seg rem
          1024, //remote
          32768, //size
          1, //queue
          GASPI_BLOCK) != GASPI_SUCCESS)
    {
      gaspi_queue_size(1, &queueSize);
      gaspi_printf (" failed with i = %d queue %u\n", j, queueSize);
      exit(-1);

    }


  }
  ASSERT(gaspi_wait(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for (i =0; i<nGPUs; i++){
    ASSERT (gaspi_segment_delete(i));
  }
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
