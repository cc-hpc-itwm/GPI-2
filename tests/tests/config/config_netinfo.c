#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_config_t default_conf;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_config_get(&default_conf));
  default_conf.net_info = 1;

  ASSERT (gaspi_config_set(default_conf));

  ASSERT (gaspi_config_get(&default_conf));

  assert ((default_conf.net_info == 1));

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
