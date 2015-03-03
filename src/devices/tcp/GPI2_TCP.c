/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2014

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
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_TCP.h"

#include "tcp_device.h"
#include "utils.h"

/* Globals */
//TODO: not really its place
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
pgaspi_dev_get_sizeof_rc()
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
  return 0;
}

int
pgaspi_dev_init_core()
{
  int c;
  
  memset (&glb_gaspi_ctx_tcp, 0, sizeof (gaspi_tcp_ctx));
  
  /* start virtual device (thread) */
  if(pthread_create(&tcp_dev_thread, NULL, tcp_virt_dev, NULL) != 0)
    {
      printf("Failed to open (virtual) device\n");
      return -1;
    }

  /* make sure all device threads are connected */
  while(!tcp_dev_connected_to_all)
    {
      __asm__ ( "pause;" );
    }

  //TODO: more info (IP, hostname...)?
  if (glb_gaspi_cfg.net_info)
    {
      gaspi_printf ("<<<<<<<<<<<<<<<< TCP-info >>>>>>>>>>>>>>>>>>>\n");
      gaspi_printf("Using TCP - Experimental stuff\n");
    }

  /* Passive channel (SRQ) */
  glb_gaspi_ctx_tcp.srqP = connect2port("localhost", PORT + glb_gaspi_ctx.localSocket, CONN_TIMEOUT);
  if(glb_gaspi_ctx_tcp.srqP == -1)
    {
      gaspi_dev_print_error("Failed to create passive channel connection");
      return -1;
    }

  glb_gaspi_ctx_tcp.channelP = tcp_dev_create_passive_channel();
  if(glb_gaspi_ctx_tcp.channelP == NULL)
    {
      gaspi_dev_print_error("Failed to create passive channel.");
      return -1;
    }
  
  /* Completion Queues */
  glb_gaspi_ctx_tcp.scqGroups = tcp_dev_create_cq(glb_gaspi_cfg.queue_depth, NULL);
  if(glb_gaspi_ctx_tcp.scqGroups == NULL)
    {
      gaspi_dev_print_error("Failed to create groups send completion queue.");
      return -1;
    }
  
  glb_gaspi_ctx_tcp.rcqGroups = tcp_dev_create_cq(glb_gaspi_cfg.queue_depth, NULL);
  if(glb_gaspi_ctx_tcp.rcqGroups == NULL)
    {
      gaspi_dev_print_error("Failed to create groups receive completion queue.");
      return -1;
    }

  glb_gaspi_ctx_tcp.scqP = tcp_dev_create_cq(glb_gaspi_cfg.queue_depth, NULL);
  if(glb_gaspi_ctx_tcp.scqP == NULL)
    {
      gaspi_dev_print_error("Failed to create passive send completion queue.");
      return -1;
    }

  glb_gaspi_ctx_tcp.rcqP = tcp_dev_create_cq(glb_gaspi_cfg.queue_depth, glb_gaspi_ctx_tcp.channelP);
  if(glb_gaspi_ctx_tcp.rcqP == NULL)
    {
      gaspi_dev_print_error("Failed to create passive recv completion queue.");
      return -1;
    }

  for(c = 0; c < glb_gaspi_cfg.queue_num; c++)
    {
      glb_gaspi_ctx_tcp.scqC[c] = tcp_dev_create_cq(glb_gaspi_cfg.queue_depth, NULL);
      if(glb_gaspi_ctx_tcp.scqC[c] == NULL)
	{
	  gaspi_dev_print_error("Failed to create IO completion queue.");
	  return -1;
	}
    }

  /* Queues (QPs) */
  glb_gaspi_ctx_tcp.qpGroups = tcp_dev_create_queue(glb_gaspi_ctx_tcp.scqGroups,
						    glb_gaspi_ctx_tcp.rcqGroups);
  if(glb_gaspi_ctx_tcp.qpGroups == NULL)
    {
      gaspi_dev_print_error("Failed to create queue for groups.");
      return -1;
    }

  for(c = 0; c < glb_gaspi_cfg.queue_num; c++)
    {
      glb_gaspi_ctx_tcp.qpC[c] = tcp_dev_create_queue( glb_gaspi_ctx_tcp.scqC[c],
						       glb_gaspi_ctx_tcp.rcqC[c]);
      if(glb_gaspi_ctx_tcp.qpC[c] == NULL)
	{
	  gaspi_dev_print_error("Failed to create queue %d for IO.", c);
	  return -1;
	}
    }
  
  glb_gaspi_ctx_tcp.qpP = tcp_dev_create_queue( glb_gaspi_ctx_tcp.scqP,
						glb_gaspi_ctx_tcp.rcqP);
  if(glb_gaspi_ctx_tcp.qpP == NULL)
    {
      gaspi_dev_print_error("Failed to create queue for passive.");
      return -1;
    }

  return 0;
}

int
pgaspi_dev_cleanup_core()
{
  int i, c;

  tcp_dev_destroy_queue(glb_gaspi_ctx_tcp.qpGroups);
  tcp_dev_destroy_queue(glb_gaspi_ctx_tcp.qpP);

  for(c = 0; c < glb_gaspi_cfg.queue_num; c++)
    {
      tcp_dev_destroy_queue(glb_gaspi_ctx_tcp.qpC[c]);
    }

  if(glb_gaspi_ctx_tcp.srqP)
    {
      if(close(glb_gaspi_ctx_tcp.srqP) < 0)
	{
	  gaspi_dev_print_error("Failed to close srqP.");
	}
    }
  
  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.scqGroups);
  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.rcqGroups);
  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.scqP);
  tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.rcqP);

  for(c = 0; c < glb_gaspi_cfg.queue_num; c++)
    {
      tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.scqC[c]);
      tcp_dev_destroy_cq(glb_gaspi_ctx_tcp.rcqC[c]);
    }
  
  for(i = 0; i < GASPI_MAX_MSEGS; i++)
    {
      if(glb_gaspi_ctx.rrmd[i] != NULL)
	{
	  if(glb_gaspi_ctx.rrmd[i][glb_gaspi_ctx.rank].size)
	    {
	      if(glb_gaspi_ctx.rrmd[i][glb_gaspi_ctx.rank].buf)
		{
		  free (glb_gaspi_ctx.rrmd[i][glb_gaspi_ctx.rank].buf);
		}
	      
	      glb_gaspi_ctx.rrmd[i][glb_gaspi_ctx.rank].buf = NULL;
	    }
	  
	  if(glb_gaspi_ctx.rrmd[i])
	    {
	      free (glb_gaspi_ctx.rrmd[i]);
	    }
	  
	  glb_gaspi_ctx.rrmd[i] = NULL;
	}
    }
  
  if(glb_gaspi_ctx_tcp.channelP)
    {
      tcp_dev_destroy_passive_channel(glb_gaspi_ctx_tcp.channelP);
    }

/*   if(tcp_dev_stop_device(glb_gaspi_ctx_tcp.qs_handle)) */
/*     { */
/*       printf("Failed to stop device\n"); */
/*       return -1; */
/*     } */

/*   if(pthread_join(tcp_dev_thread, NULL) != 0) */
/*     return -1; */
  
  return 0;
}

