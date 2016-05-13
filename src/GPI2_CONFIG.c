/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2016

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

gaspi_config_t glb_gaspi_cfg = {
  1,				//logout
  12121,                        //sn port
  0,				//netinfo
  -1,				//netdev
  0,				//mtu
  1,				//port check
  0,				//user selected network
#ifdef GPI2_DEVICE_IB
  GASPI_IB,
#else                           //network type
  GASPI_ETHERNET,
#endif
  1024,				//queue depth
  8,				//queue count
  GASPI_MAX_GROUPS,		//group_max;
  GASPI_MAX_MSEGS,		//segment_max;
  GASPI_MAX_TSIZE_C,		//transfer_size_max;
  GASPI_MAX_NOTIFICATION,	//notification_num;
  1024,				//passive_queue_size_max;
  GASPI_MAX_TSIZE_P,		//passive_transfer_size_max;
  NEXT_OFFSET,			//allreduce_buf_size;
  255,				//allreduce_elem_max;
  GASPI_TOPOLOGY_STATIC         //build_infrastructure;
};

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
  glb_gaspi_cfg.port_check = nconf.port_check;

#ifdef GPI2_DEVICE_IB
  if( GASPI_ETHERNET == nconf.network )
#elif GPI2_DEVICE_TCP
    if( GASPI_ETHERNET != nconf.network )
#endif
      {
	gaspi_print_error("Invalid value for parameter network");
	return GASPI_ERR_CONFIG;
      }
  glb_gaspi_cfg.network = nconf.network;
  glb_gaspi_cfg.user_net = 1;

  if( nconf.netdev_id > 1 )
    {
      gaspi_print_error("Invalid value for parameter netdev_id");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.netdev_id = nconf.netdev_id;

  if( nconf.queue_num > GASPI_MAX_QP || nconf.queue_num < 1 )
    {
      gaspi_print_error("Invalid value for parameter queue_num (min=1 and max=GASPI_MAX_QP");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.queue_num = nconf.queue_num;

  if( nconf.queue_depth > GASPI_MAX_QSIZE || nconf.queue_depth < 1 )
    {
      gaspi_print_error("Invalid value for parameter queue_depth (min=1 and max=GASPI_MAX_QSIZE");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.queue_depth = nconf.queue_depth;

  if( nconf.mtu == 0 || nconf.mtu == 1024 || nconf.mtu == 2048 || nconf.mtu == 4096 )
    {
      glb_gaspi_cfg.mtu = nconf.mtu;
    }
  else
    {
      gaspi_print_error("Invalid value for parameter mtu (supported: 1024, 2048, 4096)");
      return GASPI_ERR_CONFIG;
    }

  if( nconf.sn_port < 1024 || nconf.sn_port > 65536 )
    {
      gaspi_print_error("Invalid value for parameter sn_port ( from 1024 to 65536)");
      return GASPI_ERR_CONFIG;
    }

  glb_gaspi_cfg.sn_port = nconf.sn_port;

  glb_gaspi_cfg.net_info = nconf.net_info;
  glb_gaspi_cfg.logger = nconf.logger;
  glb_gaspi_cfg.port_check = nconf.port_check;

  return GASPI_SUCCESS;
}
