#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <test_utils.h>
#include <GASPI_GPU.h>

int main(int argc, char *argv[])
{
  gaspi_config_t conf;
  gaspi_gpu_t gpus[8]; 
  gaspi_gpu_num nGPUs;

  ASSERT(gaspi_config_get(&conf));
  conf.mtu = 2048;
  conf.build_infrastructure = 0;

  ASSERT(gaspi_config_set(conf));
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t rank, num;


  ASSERT (gaspi_proc_rank(&rank));
  ASSERT (gaspi_proc_num(&num));

  ASSERT (gaspi_init_GPUs());
  ASSERT (gaspi_number_of_GPUs(&nGPUs));
  ASSERT (gaspi_GPU_ids(gpus));

  gaspi_printf("Hello from rank %d of %d -> %d with %d GPUs:\n", 
	       rank, num, (rank + 1 ) % num, nGPUs );
  int i;

  for (i = 0; i < gpus; i++)
    gaspi_printf("device with Id %d \n", gpus[i]);
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
