#include <stdio.h>
#include <stdlib.h>
#include <GASPI_GPU.h>
#include <test_utils.h>
#include <cuda.h>
//alloc GPU segemnts for all GPUs
//and then registe them with all nodes
//then delete them
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
    ASSERT (gaspi_segment_alloc(i, 1024, GASPI_MEM_INITIALIZED|GASPI_MEM_GPU));
  }

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));


  for (i = 0; i < nprocs; i++)
  {


    if(i == rank)
      continue;

    for (j =0; j<nGPUs; j++){
      ASSERT( gaspi_segment_register(j, i, GASPI_BLOCK));
      //sleep(1);
    }
  }
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for (i =0; i<nGPUs; i++){
    ASSERT (gaspi_segment_delete(i));
  }
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
