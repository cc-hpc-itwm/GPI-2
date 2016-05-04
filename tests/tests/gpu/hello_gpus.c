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
  gaspi_gpu_id_t gpus[8]; 
  gaspi_number_t nGPUs;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t rank, num;
  ASSERT (gaspi_proc_rank(&rank));
  ASSERT (gaspi_proc_num(&num));

  ASSERT (gaspi_gpu_init());
  ASSERT (gaspi_gpu_number(&nGPUs));
  ASSERT (gaspi_gpu_ids(gpus));

  printf("Hello from rank %d of %d with %d GPUs:\n", 
	 rank, num, nGPUs );
  int i;
  for (i = 0; i < gpus; i++)
    {
      gaspi_printf("device with Id %d \n", gpus[i]);
    }

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
