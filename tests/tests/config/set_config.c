#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_config_t default_conf;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_config_get(&default_conf));

  //MTU
  default_conf.mtu = 5000;
  EXPECT_FAIL (gaspi_config_set(default_conf));

  default_conf.mtu = 4096;
  ASSERT (gaspi_config_set(default_conf));
  
  default_conf.netdev_id = 2;
  EXPECT_FAIL (gaspi_config_set(default_conf));
 
  //  GASPI_NETWORK_TYPE
  default_conf.netdev_id = 0;
  default_conf.network = GASPI_IB;
  default_conf.mtu = 2048;
  ASSERT (gaspi_config_set(default_conf));


  //############################################//
  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  //Don't allow setup after init
  default_conf.mtu = 2048;
  EXPECT_FAIL (gaspi_config_set(default_conf));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
