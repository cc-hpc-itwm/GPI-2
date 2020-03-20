#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

// Test to check if user_net parameter in config is set or not set
// depending on whether default network is set.  Currently, only when
// network == GASPI_ROCE we expect the user_net to be set.

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  gaspi_config_t default_conf;

  ASSERT (gaspi_config_get (&default_conf));

  assert (default_conf.user_net ==
          (default_conf.network == GASPI_ROCE ? 1 : 0));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
