/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2017

This file is part of GPI-2.

GPI-2 is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License
version 3 as published by the Free Software Foundation.

GPI-2 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GPI-2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "GPI2.h"
#include "GPI2_Utility.h"

static const char* gaspi_network_str [] =
  {
    [GASPI_IB] = "GASPI_IB",
    [GASPI_ROCE] = "GASPI_ROCE",
    [GASPI_ETHERNET] = "GASPI_ETHERNET",
    [GASPI_GEMINI] = "GASPI_GEMINI",
    [GASPI_ARIES] = "GASPI_ARIES"
  };

gaspi_config_t glb_gaspi_cfg = {
  1,				//logout
  12121,                        //sn port
  0,				//netinfo
  0,				//user selected network
  1,                            //sn persistent
  30000,                        //sn timeout
#ifdef GPI2_DEVICE_IB
  {
    GASPI_IB,
    {
      {
	-1,                     //netdev
	0,			//mtu
	1,                      //port check
      },
      {
	0                       //port to use
      }
    }
  },
  GASPI_IB,                     //network type
#else
  {
    GASPI_ETHERNET,
    {
      {
	-1,                     //netdev
	0,			//mtu
	1,                      //port check
      },
      {
	19000                   //port to use
      }
    }
  },
  GASPI_ETHERNET,               //network type
#endif
  1024,				//queue size max
  8,				//queue count
  GASPI_MAX_GROUPS,		//group_max;
  GASPI_MAX_MSEGS,		//segment_max;
  GASPI_MAX_TSIZE_C,		//transfer_size_max;
  GASPI_MAX_NOTIFICATION,	//notification_num;
  1024,				//passive_queue_size_max;
  GASPI_MAX_TSIZE_P,		//passive_transfer_size_max;
  (255*sizeof(unsigned long)),  //allreduce_buf_size;
  255,				//allreduce_elem_max;
  GASPI_TOPOLOGY_STATIC         //build_infrastructure;
};

static void
pgaspi_print_config(gaspi_config_t * const config)
{
  printf(" logger %u\nsn_port %u\nnet_info %u\n \
user_net %u\nsn persistent %d\nnetwork %d\nqueue_size_max %u\nqueue_num %u\n \
group_max %d\nsegment_max %d\ntransfer_size_max %lu\nnotification_num %u\n \
passive_queue_size_max %u\npassive_transfer_size_max %u\nallreduce_buf_size %lu\n \
allreduce_elem_max %u\nbuild_infrastructure %d\n",
	 config->logger,
	 config->sn_port,
	 config->net_info,
	 config->user_net,
	 config->sn_persistent,
	 config->network,
	 config->queue_size_max,
	 config->queue_num,
	 config->group_max,
	 config->segment_max,
	 config->transfer_size_max,
	 config->notification_num,
	 config->passive_queue_size_max,
	 config->passive_transfer_size_max,
	 config->allreduce_buf_size,
	 config->allreduce_elem_max,
	 config->build_infrastructure);

  if( config->network == GASPI_IB )
    {
      printf("Device-dependent\n: netdev_id %d\n mtu %u\n port_check %u\n",
	     config->dev_config.params.ib.netdev_id,
	     config->dev_config.params.ib.mtu,
	     config->dev_config.params.ib.port_check);
    }
  if( config->network == GASPI_ETHERNET )
    {
      printf("Device-dependent:\n: port %u\n",
	     config->dev_config.params.tcp.port);
    }
}

#pragma weak gaspi_config_get = pgaspi_config_get
gaspi_return_t
pgaspi_config_get (gaspi_config_t * const config)
{
  gaspi_verify_null_ptr(config);

  memcpy (config, &glb_gaspi_cfg, sizeof (gaspi_config_t));

  return GASPI_SUCCESS;
}

#pragma weak gaspi_config_set = pgaspi_config_set
gaspi_return_t
pgaspi_config_set (const gaspi_config_t nconf)
{
  gaspi_verify_setup("gaspi_config_set");

  glb_gaspi_cfg.net_info = nconf.net_info;
  glb_gaspi_cfg.build_infrastructure = nconf.build_infrastructure;
  glb_gaspi_cfg.logger = nconf.logger;

#ifdef GPI2_DEVICE_IB
  if( nconf.dev_config.params.ib.mtu == 0 ||
      nconf.dev_config.params.ib.mtu == 1024 ||
      nconf.dev_config.params.ib.mtu == 2048 ||
      nconf.dev_config.params.ib.mtu == 4096 )
    {
      glb_gaspi_cfg.dev_config.params.ib.mtu = nconf.dev_config.params.ib.mtu;
    }
  else
    {
      gaspi_print_error("Invalid value for parameter mtu (supported: 1024, 2048, 4096)");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.dev_config.params.ib.port_check = nconf.dev_config.params.ib.port_check;

  glb_gaspi_cfg.dev_config.params.ib.netdev_id = nconf.dev_config.params.ib.netdev_id;
  if( nconf.dev_config.params.ib.netdev_id > 1 )
    {
      gaspi_print_error("Invalid value for parameter netdev_id");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.dev_config.params.ib.port_check = nconf.dev_config.params.ib.port_check;
  if( GASPI_ETHERNET == nconf.network )
#elif GPI2_DEVICE_TCP
    if( GASPI_ETHERNET != nconf.network )
#endif
      {
	gaspi_print_error("Invalid value for parameter network (%s)", gaspi_network_str[nconf.network]);
	return GASPI_ERR_CONFIG;
      }
  glb_gaspi_cfg.network = nconf.network;
  glb_gaspi_cfg.user_net = 1;



  if( nconf.queue_num > GASPI_MAX_QP || nconf.queue_num < 1 )
    {
      gaspi_print_error("Invalid value for parameter queue_num (min=1 and max=GASPI_MAX_QP");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.queue_num = nconf.queue_num;

  if( nconf.queue_size_max > GASPI_MAX_QSIZE || nconf.queue_size_max < 1 )
    {
      gaspi_print_error("Invalid value for parameter queue_size_max (min=1 and max=GASPI_MAX_QSIZE");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.queue_size_max = nconf.queue_size_max;

  if( nconf.sn_port < 1024 || nconf.sn_port > 65536 )
    {
      gaspi_print_error("Invalid value for parameter sn_port ( from 1024 to 65536)");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.sn_port = nconf.sn_port;
  glb_gaspi_cfg.sn_persistent = nconf.sn_persistent;

  if( nconf.sn_timeout < 0 )
    {
      gaspi_print_error("Invalid value for parameter sn_timeout ( must be > 0)");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.sn_timeout = nconf.sn_timeout;

  glb_gaspi_cfg.net_info = nconf.net_info;
  glb_gaspi_cfg.logger = nconf.logger;

  glb_gaspi_cfg.allreduce_elem_max = nconf.allreduce_elem_max;

  return GASPI_SUCCESS;
}
