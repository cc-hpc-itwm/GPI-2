#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  gaspi_config_t default_conf;

  ASSERT (gaspi_config_get (&default_conf));

  //MTU
  if (default_conf.network == GASPI_IB)
  {

    default_conf.dev_config.params.ib.mtu = 5000;
    EXPECT_FAIL (gaspi_config_set (default_conf));

    default_conf.dev_config.params.ib.mtu = 4096;
    ASSERT (gaspi_config_set (default_conf));

    default_conf.dev_config.params.ib.netdev_id = 5;
    EXPECT_FAIL (gaspi_config_set (default_conf));

    default_conf.dev_config.params.ib.netdev_id = 0;
    default_conf.dev_config.params.ib.mtu = 2048;
    ASSERT (gaspi_config_set (default_conf));

    ASSERT (gaspi_config_get (&default_conf));
    assert (default_conf.dev_config.params.ib.netdev_id == 0);
    assert (default_conf.dev_config.params.ib.mtu == 2048);
  }
  else if (default_conf.network == GASPI_ETHERNET)
  {
    default_conf.dev_config.params.tcp.port = 30000;
  }

  default_conf.sn_port = 21212;
  ASSERT (gaspi_config_set (default_conf));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  //Don't allow setup after init
  default_conf.sn_port = 20000;
  EXPECT_FAIL (gaspi_config_set (default_conf));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
