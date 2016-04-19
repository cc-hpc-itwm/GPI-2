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
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_SN.h"
#include "GPI2_TCP.h"
#include "GPI2_Utility.h"
#include "tcp_device.h"

extern gaspi_config_t glb_gaspi_cfg;

/* TODO: empty functions smell */
inline char *
pgaspi_dev_get_rrcd(int rank)
{
  return NULL;
}


inline char *
pgaspi_dev_get_lrcd(int rank)
{
  return NULL;
}

inline size_t
pgaspi_dev_get_sizeof_rc(void)
{
  return 0;
}

int
pgaspi_dev_create_endpoint(const int i)
{
  return 0;
}

int
pgaspi_dev_disconnect_context(const int i)
{
  return 0;
}

int
pgaspi_dev_connect_context(const int i)
{
  return tcp_dev_connect_to(i);
}

int
pgaspi_dev_comm_queue_connect(const unsigned short q, const int i)
{
  return 0;
}

int
pgaspi_dev_comm_queue_delete(const unsigned int id)
{
  tcp_dev_destroy_queue(glb_gaspi_ctx_tcp.qpC[id]);
  glb_gaspi_ctx_tcp.qpC[id] = NULL;

  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.scqC[id]);
  glb_gaspi_ctx_tcp.scqC[id] = NULL;

  return 0;
}

int
pgaspi_dev_comm_queue_create(const unsigned int id, const unsigned short remote_node)
{
  if( glb_gaspi_ctx_tcp.scqC[id] == NULL)
    {
      glb_gaspi_ctx_tcp.scqC[id] = tcp_dev_create_cq(glb_gaspi_cfg.queue_depth, NULL);
      if(glb_gaspi_ctx_tcp.scqC[id] == NULL)
	{
	  gaspi_print_error("Failed to create IO completion queue.");
	  return -1;
	}
    }

  if( glb_gaspi_ctx_tcp.qpC[id] == NULL)
    {
      glb_gaspi_ctx_tcp.qpC[id] = tcp_dev_create_queue( glb_gaspi_ctx_tcp.scqC[id], NULL);
      if(glb_gaspi_ctx_tcp.qpC[id] == NULL)
	{
	  gaspi_print_error("Failed to create queue %d for IO.", id);
	  return -1;
	}
    }

  return 0;
}

static void
pgaspi_dev_print_info()
{
  gaspi_printf("<<<<<<<<<<<<<<<< TCP-info >>>>>>>>>>>>>>>>>>>\n");
  gaspi_printf("  Hostname: %s\n", gaspi_get_hn(glb_gaspi_ctx.rank));

  char* ip = tcp_dev_get_local_ip();
  if( ip != NULL )
    {
      tcp_dev_get_local_if(ip);
      gaspi_printf("  %-8s: %s\n", tcp_dev_get_local_if(ip), ip );
    }
  else
    {
      gaspi_printf("  Failed to retrieve more info\n");
    }
  gaspi_printf("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>\n");
}

int
pgaspi_dev_init_core(gaspi_config_t *gaspi_cfg)
{
  unsigned int c;
  
  memset (&glb_gaspi_ctx_tcp, 0, sizeof (gaspi_tcp_ctx));
  
  /* start virtual device (thread) */
  if(pthread_create(&tcp_dev_thread, NULL, tcp_virt_dev, NULL) != 0)
    {
      gaspi_print_error("Failed to open (virtual) device.");
      return -1;
    }

  /* user did not choose so we set the network type */
  if( !gaspi_cfg->user_net )
    {
      gaspi_cfg->network = GASPI_ETHERNET;
    }

  if (gaspi_cfg->net_info)
    {
      pgaspi_dev_print_info();
    }

  /* Passive channel (SRQ) */
  glb_gaspi_ctx_tcp.srqP = gaspi_sn_connect2port("localhost", TCP_DEV_PORT + glb_gaspi_ctx.localSocket, CONN_TIMEOUT);
  if(glb_gaspi_ctx_tcp.srqP == -1)
    {
      gaspi_print_error("Failed to create passive channel connection");
      return -1;
    }

  glb_gaspi_ctx_tcp.channelP = tcp_dev_create_passive_channel();
  if(glb_gaspi_ctx_tcp.channelP == NULL)
    {
      gaspi_print_error("Failed to create passive channel.");
      return -1;
    }
  
  /* Completion Queues */
  glb_gaspi_ctx_tcp.scqGroups = tcp_dev_create_cq(gaspi_cfg->queue_depth, NULL);
  if(glb_gaspi_ctx_tcp.scqGroups == NULL)
    {
      gaspi_print_error("Failed to create groups send completion queue.");
      return -1;
    }
  
  glb_gaspi_ctx_tcp.rcqGroups = tcp_dev_create_cq(gaspi_cfg->queue_depth, NULL);
  if(glb_gaspi_ctx_tcp.rcqGroups == NULL)
    {
      gaspi_print_error("Failed to create groups receive completion queue.");
      return -1;
    }

  glb_gaspi_ctx_tcp.scqP = tcp_dev_create_cq(gaspi_cfg->queue_depth, NULL);
  if(glb_gaspi_ctx_tcp.scqP == NULL)
    {
      gaspi_print_error("Failed to create passive send completion queue.");
      return -1;
    }

  glb_gaspi_ctx_tcp.rcqP = tcp_dev_create_cq(gaspi_cfg->queue_depth, glb_gaspi_ctx_tcp.channelP);
  if(glb_gaspi_ctx_tcp.rcqP == NULL)
    {
      gaspi_print_error("Failed to create passive recv completion queue.");
      return -1;
    }

  for(c = 0; c < gaspi_cfg->queue_num; c++)
    {
      glb_gaspi_ctx_tcp.scqC[c] = tcp_dev_create_cq(gaspi_cfg->queue_depth, NULL);
      if(glb_gaspi_ctx_tcp.scqC[c] == NULL)
	{
	  gaspi_print_error("Failed to create IO completion queue.");
	  return -1;
	}
    }

  /* Queues (QPs) */
  glb_gaspi_ctx_tcp.qpGroups = tcp_dev_create_queue(glb_gaspi_ctx_tcp.scqGroups,
						    glb_gaspi_ctx_tcp.rcqGroups);
  if(glb_gaspi_ctx_tcp.qpGroups == NULL)
    {
      gaspi_print_error("Failed to create queue for groups.");
      return -1;
    }

  for(c = 0; c < gaspi_cfg->queue_num; c++)
    {
      glb_gaspi_ctx_tcp.qpC[c] = tcp_dev_create_queue( glb_gaspi_ctx_tcp.scqC[c],
						       NULL);
      if(glb_gaspi_ctx_tcp.qpC[c] == NULL)
	{
	  gaspi_print_error("Failed to create queue %d for IO.", c);
	  return -1;
	}
    }
  
  glb_gaspi_ctx_tcp.qpP = tcp_dev_create_queue( glb_gaspi_ctx_tcp.scqP,
						glb_gaspi_ctx_tcp.rcqP);
  if(glb_gaspi_ctx_tcp.qpP == NULL)
    {
      gaspi_print_error("Failed to create queue for passive.");
      return -1;
    }

  gaspi_tcp_dev_status_t _dev_status = gaspi_tcp_dev_status_get();

  while( GASPI_TCP_DEV_STATUS_DOWN == _dev_status )
    {
      gaspi_delay();
      _dev_status = gaspi_tcp_dev_status_get();
    }

  if( GASPI_TCP_DEV_STATUS_FAILED == _dev_status )
    return -1;

  return 0;
}

int
pgaspi_dev_cleanup_core(gaspi_config_t *gaspi_cfg)
{
  int i, s;
  void *res;
  unsigned int c;

  tcp_dev_stop_device();

  /* Destroy posting queues and associated channels */
  tcp_dev_destroy_queue(glb_gaspi_ctx_tcp.qpGroups);
  tcp_dev_destroy_queue(glb_gaspi_ctx_tcp.qpP);

  for(c = 0; c < gaspi_cfg->queue_num; c++)
    {
      tcp_dev_destroy_queue(glb_gaspi_ctx_tcp.qpC[c]);
    }

  if(glb_gaspi_ctx_tcp.srqP)
    {
      if(close(glb_gaspi_ctx_tcp.srqP) < 0)
	{
	  gaspi_print_error("Failed to close srqP.");
	}
    }
  
  if(glb_gaspi_ctx_tcp.channelP)
    {
      tcp_dev_destroy_passive_channel(glb_gaspi_ctx_tcp.channelP);
    }

  s = pthread_join(tcp_dev_thread, &res);
  if (s != 0)
    {
      gaspi_print_error("Failed to wait device.");
    }

  /* Now we can destroy the resources for completion and incoming data */
  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.scqGroups);
  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.rcqGroups);
  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.scqP);
  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.rcqP);

  for(c = 0; c < gaspi_cfg->queue_num; c++)
    {
      tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.scqC[c]);
    }

  //TODO: why is this here?
  for(i = 0; i < GASPI_MAX_MSEGS; i++)
    {
      if(glb_gaspi_ctx.rrmd[i] != NULL)
	{
	  if(glb_gaspi_ctx.rrmd[i][glb_gaspi_ctx.rank].size)
	    {
	      free (glb_gaspi_ctx.rrmd[i][glb_gaspi_ctx.rank].notif_spc.buf);
	      glb_gaspi_ctx.rrmd[i][glb_gaspi_ctx.rank].notif_spc.buf = NULL;
	      glb_gaspi_ctx.rrmd[i][glb_gaspi_ctx.rank].data.buf = NULL;
	    }
	  
	  free (glb_gaspi_ctx.rrmd[i]);
	  glb_gaspi_ctx.rrmd[i] = NULL;
	}
    }

  return 0;
}
