#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_config_t default_conf;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_config_get(&default_conf));

  gaspi_printf("logger %u\nnet_info %u\nnetdev_info %u\nmtu %u\nport_check %u\nuser_net %u\nnet_type %u\n",
	       default_conf.logger,
	       default_conf.net_info,
	       default_conf.netdev_id,
	       default_conf.mtu,
	       default_conf.port_check,
	       default_conf.user_net,
	       default_conf.net_typ);
  
  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
