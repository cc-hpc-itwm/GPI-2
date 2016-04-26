#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <test_utils.h>
int main(int argc, char *argv[])
{
  gaspi_config_t conf;
  ASSERT(gaspi_config_get(&conf));
  conf.mtu = 4096;
  conf.queue_num = 1;
  ASSERT(gaspi_config_set(conf));
 
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  gaspi_rank_t rank, num;

  ASSERT (gaspi_proc_rank(&rank));
  ASSERT (gaspi_proc_num(&num));
  
  printf("Hello from rank %d of %d\n", rank, num);

  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
