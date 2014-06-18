#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_config_t default_conf;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_config_get(&default_conf));

  printf("Using port 21212\n");
  
  default_conf.sn_port = 21212;

  ASSERT (gaspi_config_set(default_conf));


  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
