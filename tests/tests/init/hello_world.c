#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <test_utils.h>
int main(int argc, char *argv[])
{
  gaspi_config_t conf;
  ASSERT(gaspi_config_get(&conf));
  conf.mtu = 2048;
  conf.build_infrastructure = 0;
  ASSERT(gaspi_config_set(conf));
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  gaspi_rank_t rank, num;

  ASSERT (gaspi_proc_rank(&rank));
  ASSERT (gaspi_proc_num(&num));
  
  gaspi_printf("Hello from rank %d of %d -> %d\n", 
	       rank, num, (rank + 1 ) % num );

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
