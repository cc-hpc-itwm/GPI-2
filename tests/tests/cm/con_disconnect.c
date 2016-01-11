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

  TSUITE_INIT(argc, argv);

  gaspi_config_get(&conf);
  conf.build_infrastructure = GASPI_TOPOLOGY_NONE;
  gaspi_config_set(conf);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  gaspi_rank_t rank, num;

  ASSERT (gaspi_proc_rank(&rank));
  ASSERT (gaspi_proc_num(&num));

  int i;
  for (i = 0; i < num; i++)
    {
      printf("Rank %u: connect to %d\n", rank, i);
      ASSERT(gaspi_connect(i, GASPI_BLOCK));
    }
  for (i = 0; i < num; i++)
    {
      printf("Rank %u: disconnect to %d\n", rank, i);
      gaspi_disconnect(i, GASPI_BLOCK);
    }

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  printf("Finished!\n");
  
  return EXIT_SUCCESS;
}
