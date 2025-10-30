/*
Copyright (c) Fraunhofer ITWM, 2013-2025

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

#include <string.h>
#include "GASPI_types.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

#define GASPI_MAX_GROUPS  (32)
#define GASPI_MAX_MSEGS   (255)
#define GASPI_MAX_QSIZE   (4096)
#define GASPI_MAX_TSIZE_C (1ul<<30ul)
#define GASPI_MAX_TSIZE_P ((1ul<<16ul)-1ul)
#define GASPI_DEFAULT_QSIZE (1024)
#define GASPI_DEFAULT_ALLREDUCE_ELEM_MAX 255
#define GASPI_DEFAULT_ALLREDUCE_BUF_SIZE        \
  (GASPI_DEFAULT_ALLREDUCE_ELEM_MAX * sizeof (unsigned long))

extern gaspi_context_t glb_gaspi_ctx;

gaspi_config_t glb_gaspi_cfg =
{
  1,                                //logout
  12121,                            //sn port
  0,                                //netinfo
  0,                                //user selected network
  1,                                //sn persistent
  30000,                            //sn timeout
#ifdef GPI2_DEVICE_IB
  {
    GASPI_IB,
    {
      {
        -1,                         //netdev
        0,                          //mtu
        1,                          //port check
      },
      {
        0                           //port to use
      },
      {
        1,                          //use shmem
        0,                          //print provider info
        1,                          //progress auto
      }
    }
  },
  GASPI_IB,                         //network type
#elif GPI2_DEVICE_TCP
  {
    GASPI_ETHERNET,
    {
      {
        -1,                         //netdev
        0,                          //mtu
        1,                          //port check
      },
      {
        19000                       //port to use
      },
      {
        1,                          //use shmem
        0,                          //print provider info
        1,                          //progress auto
      }
    }
  },
  GASPI_ETHERNET,                   //network type
#elif GPI2_DEVICE_OFI
  {
    GASPI_OFI,
    {
      {
        -1,                         //netdev
        0,                          //mtu
        1,                          //port check
      },
      {
        19000                       //port to use
      },
      {
        1,                          //use shmem
        0,                          //print provider info
        1,                          //progress auto
      }
    }
  },
  GASPI_OFI,                   //network type
#endif
  GASPI_DEFAULT_QSIZE,              //queue size max
  8,                                //queue count
  GASPI_MAX_GROUPS,                 //group_max;
  GASPI_MAX_MSEGS,                  //segment_max;
  GASPI_MAX_TSIZE_C,                //transfer_size_max;
  GASPI_DEFAULT_MAX_NOTIFICATION,   //notification_num;
  GASPI_DEFAULT_QSIZE,              //passive_queue_size_max;
  GASPI_MAX_TSIZE_P,                //passive_transfer_size_max;
  GASPI_DEFAULT_ALLREDUCE_BUF_SIZE, //allreduce_buf_size;
  GASPI_DEFAULT_ALLREDUCE_ELEM_MAX, //allreduce_elem_max;
  GASPI_RW_LIST_ELEM_MAX, // maximum elements in lists
  GASPI_TOPOLOGY_STATIC,            //build_infrastructure;
  NULL                              //user_defined
};

#pragma weak gaspi_config_get = pgaspi_config_get
gaspi_return_t
pgaspi_config_get (gaspi_config_t * const config)
{
  GASPI_VERIFY_NULL_PTR (config);

  memcpy (config, &glb_gaspi_cfg, sizeof (gaspi_config_t));

  return GASPI_SUCCESS;
}

#pragma weak gaspi_config_set = pgaspi_config_set
gaspi_return_t
pgaspi_config_set (const gaspi_config_t nconf)
{
  GASPI_VERIFY_SETUP ("gaspi_config_set");

  //GPI-2 only
  glb_gaspi_cfg.logger = nconf.logger;

  if (nconf.sn_port < 1024 || nconf.sn_port > 65536)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Invalid value for parameter sn_port (from 1024 to 65536)");
    return GASPI_ERR_CONFIG;
  }
  glb_gaspi_cfg.sn_port = nconf.sn_port;

  glb_gaspi_cfg.net_info = nconf.net_info;
  glb_gaspi_cfg.sn_persistent = nconf.sn_persistent;
  glb_gaspi_cfg.sn_timeout = nconf.sn_timeout;

#ifdef GPI2_DEVICE_IB
  if (nconf.dev_config.params.ib.mtu == 0 ||
      nconf.dev_config.params.ib.mtu == 1024 ||
      nconf.dev_config.params.ib.mtu == 2048 ||
      nconf.dev_config.params.ib.mtu == 4096)
  {
    glb_gaspi_cfg.dev_config.params.ib.mtu = nconf.dev_config.params.ib.mtu;
  }
  else
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Invalid value for parameter mtu (supported: 1024, 2048, 4096)");
    return GASPI_ERR_CONFIG;
  }

  glb_gaspi_cfg.dev_config.params.ib.port_check =
    nconf.dev_config.params.ib.port_check;

  glb_gaspi_cfg.dev_config.params.ib.netdev_id =
    nconf.dev_config.params.ib.netdev_id;
  if (nconf.dev_config.params.ib.netdev_id > 4)
  {
    GASPI_DEBUG_PRINT_ERROR ("Invalid value for parameter netdev_id");
    return GASPI_ERR_CONFIG;
  }

  glb_gaspi_cfg.dev_config.params.ib.port_check =
    nconf.dev_config.params.ib.port_check;
  if (GASPI_ETHERNET == nconf.network)
#elif GPI2_DEVICE_TCP
  if (GASPI_ETHERNET != nconf.network)
#elif GPI2_DEVICE_OFI
  if (GASPI_OFI != nconf.network)
#endif
  {
#ifdef DEBUG
    const char *gaspi_network_str[] =
      {
        [GASPI_IB] = "GASPI_IB",
        [GASPI_ROCE] = "GASPI_ROCE",
        [GASPI_ETHERNET] = "GASPI_ETHERNET",
        [GASPI_GEMINI] = "GASPI_GEMINI",
        [GASPI_ARIES] = "GASPI_ARIES"
      };
#endif
    GASPI_DEBUG_PRINT_ERROR ("Invalid value for parameter network (%s)",
                             gaspi_network_str[nconf.network]);
    return GASPI_ERR_CONFIG;
  }

#if GPI2_DEVICE_OFI
  if (nconf.dev_config.params.ofi.use_shm >= 0)
  {
    glb_gaspi_cfg.dev_config.params.ofi.use_shm =
      nconf.dev_config.params.ofi.use_shm;
  }
  else
  {
    GASPI_DEBUG_PRINT_ERROR ("Invalid value for parameter use_shm");
    return GASPI_ERR_CONFIG;
  }

  if (nconf.dev_config.params.ofi.provider_info >= 0)
  {
    glb_gaspi_cfg.dev_config.params.ofi.provider_info =
      nconf.dev_config.params.ofi.provider_info;
  }
  else
  {
    GASPI_DEBUG_PRINT_ERROR ("Invalid value for parameter provider_info");
    return GASPI_ERR_CONFIG;
  }

#endif

  //GASPI specified
  if (nconf.network != glb_gaspi_cfg.network)
  {
    glb_gaspi_cfg.network = nconf.network;

    //user has set the network
    glb_gaspi_cfg.user_net = 1;
  }

  if (nconf.queue_size_max > GASPI_MAX_QSIZE || nconf.queue_size_max < 1)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Invalid value for parameter queue_size_max (min=1 and max=%u)",
       GASPI_MAX_QSIZE);
    return GASPI_ERR_CONFIG;
  }

  glb_gaspi_cfg.queue_size_max = nconf.queue_size_max;

  if (nconf.queue_num > GASPI_MAX_QP || nconf.queue_num < 1)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Invalid value for parameter queue_num (min=1 and max=GASPI_MAX_QP");
    return GASPI_ERR_CONFIG;
  }

  glb_gaspi_cfg.queue_num = nconf.queue_num;

  glb_gaspi_cfg.group_max = nconf.group_max;

  if (nconf.segment_max > GASPI_MAX_MSEGS || nconf.segment_max < 1)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Invalid value for parameter segment_max (min=1 and max=GASPI_MAX_MSEGS)");
    return GASPI_ERR_CONFIG;
  }

  glb_gaspi_cfg.segment_max = nconf.segment_max;
  glb_gaspi_cfg.group_max = nconf.group_max;

  // TODO: the default value probably should come from the device
  if (nconf.transfer_size_max > GASPI_MAX_TSIZE_C)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Invalid value for parameter transfer_size_max (max=%lu)",
       GASPI_MAX_TSIZE_C);

    return GASPI_ERR_CONFIG;
  }
  glb_gaspi_cfg.transfer_size_max = nconf.transfer_size_max;

  glb_gaspi_cfg.notification_num = nconf.notification_num;

  if (nconf.passive_queue_size_max != GASPI_DEFAULT_QSIZE)
  {
    GASPI_PRINT_WARNING
      ("The current implementation does not consider the use of the parameter\
 passive_queue_size_max");
  }

  // TODO: the default value probably should come from the device
  if (nconf.passive_transfer_size_max > GASPI_MAX_TSIZE_P)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Invalid value (%u) for parameter passive_transfer_size_max (max=%lu)",
       nconf.passive_transfer_size_max,
       GASPI_MAX_TSIZE_P);

    return GASPI_ERR_CONFIG;
  }
  glb_gaspi_cfg.passive_transfer_size_max = nconf.passive_transfer_size_max;

  if (nconf.allreduce_buf_size != 255 * sizeof (unsigned long))
  {
    GASPI_PRINT_WARNING
      ("The current implementation does not consider the use of the parameter\
 allreduce_buf_size");
  }

  glb_gaspi_cfg.allreduce_elem_max = nconf.allreduce_elem_max;

  if (nconf.rw_list_elem_max > GASPI_RW_LIST_ELEM_MAX || nconf.rw_list_elem_max < 1)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Invalid value for parameter rw_list_elem_max (min=1 and max=GASPI_RW_LIST_ELEM_MAX)");
    return GASPI_ERR_CONFIG;
  }

  glb_gaspi_cfg.rw_list_elem_max = nconf.rw_list_elem_max;

  glb_gaspi_cfg.build_infrastructure = nconf.build_infrastructure;

  if (nconf.user_defined != NULL)
  {
    GASPI_PRINT_WARNING
      ("The current implementation does not consider the use of the parameter\
 user_defined");
  }

  return GASPI_SUCCESS;
}
