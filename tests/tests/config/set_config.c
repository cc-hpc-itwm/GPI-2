#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  gaspi_config_t default_conf;
  ASSERT (gaspi_config_get(&default_conf));

  //MTU
  default_conf.mtu = 5000;
  EXPECT_FAIL (gaspi_config_set(default_conf));

  default_conf.mtu = 4096;
  ASSERT (gaspi_config_set(default_conf));

  default_conf.netdev_id = 2;
  EXPECT_FAIL (gaspi_config_set(default_conf));

  default_conf.netdev_id = 0;
  default_conf.mtu = 2048;
  default_conf.sn_port = 21212;
  ASSERT (gaspi_config_set(default_conf));

  ASSERT (gaspi_config_get(&default_conf));
  assert(default_conf.sn_port == 21212);
  assert(default_conf.netdev_id == 0);
  assert(default_conf.mtu == 2048);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  //Don't allow setup after init
  default_conf.mtu = 2048;
  EXPECT_FAIL (gaspi_config_set(default_conf));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
