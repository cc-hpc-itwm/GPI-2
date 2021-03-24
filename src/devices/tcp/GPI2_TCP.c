/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2021

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

int
pgaspi_dev_create_endpoint (gaspi_context_t const *const GASPI_UNUSED (gctx),
                            const int GASPI_UNUSED (i),
                            void **info,
                            void **remote_info,
                            size_t * info_size)
{
  *info = NULL;
  *remote_info = NULL;
  *info_size = 0;

  return 0;
}

//TODO:
int
pgaspi_dev_disconnect_context (gaspi_context_t * const GASPI_UNUSED (gctx),
                               const int GASPI_UNUSED (i))
{
  return 0;
}

int
pgaspi_dev_connect_context (gaspi_context_t const *const gctx,
                            const int i)
{
  return tcp_dev_connect_to (i, pgaspi_gethostname (i),
                             gctx->config->dev_config.params.tcp.port +
                             gctx->poff[i]);
}

int
pgaspi_dev_comm_queue_connect (gaspi_context_t const *const GASPI_UNUSED (gctx),
                               const unsigned short GASPI_UNUSED (q),
                               const int GASPI_UNUSED (i))
{
  return 0;
}

int
pgaspi_dev_comm_queue_delete (gaspi_context_t const *const gctx,
                              const unsigned int id)
{
  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  tcp_dev_destroy_queue (tcp_dev_ctx->qpC[id]);
  tcp_dev_ctx->qpC[id] = NULL;

  tcp_dev_destroy_cq (tcp_dev_ctx->scqC[id]);
  tcp_dev_ctx->scqC[id] = NULL;

  return 0;
}

int
pgaspi_dev_comm_queue_create (gaspi_context_t const *const gctx,
                              const unsigned int id,
                              const unsigned short GASPI_UNUSED (remote_node))
{
  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  if (tcp_dev_ctx->scqC[id] == NULL)
  {
    tcp_dev_ctx->scqC[id] =
      tcp_dev_create_cq (gctx->config->queue_size_max, NULL);
    if (tcp_dev_ctx->scqC[id] == NULL)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create IO completion queue.");
      return -1;
    }
  }

  if (tcp_dev_ctx->qpC[id] == NULL)
  {
    tcp_dev_ctx->qpC[id] = tcp_dev_create_queue (tcp_dev_ctx->scqC[id], NULL);
    if (tcp_dev_ctx->qpC[id] == NULL)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create queue %d for IO.", id);
      return -1;
    }
  }

  return 0;
}

int
pgaspi_dev_test_queue (gaspi_context_t const *const gctx,
                                 const unsigned int id)
{
  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *)gctx->device->ctx;

  if (tcp_dev_ctx->qpC[id] == NULL)
  {
    return GASPI_ERR_INV_QUEUE;
  }

  return 0;
}

static void
pgaspi_tcp_dev_print_info (gaspi_context_t const *const gctx)
{
  gaspi_printf ("<<<<<<<<<<<<<<<< TCP-info >>>>>>>>>>>>>>>>>>>\n");
  gaspi_printf ("  Hostname: %s\n", pgaspi_gethostname (gctx->rank));

  char *ip = tcp_dev_get_local_ip (pgaspi_gethostname (gctx->rank));

  if (ip != NULL)
  {
    char *iface = tcp_dev_get_local_if (ip);

    gaspi_printf ("  %-8s: %s\n", iface, ip);
    free (iface);
  }
  else
  {
    gaspi_printf ("  Failed to retrieve more info\n");
  }
  gaspi_printf ("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>\n");
}

int
pgaspi_dev_init_core (gaspi_context_t * const gctx)
{
  gctx->device = calloc (1, sizeof (gctx->device));
  if (NULL == gctx->device)
  {
    return -1;
  }

  gctx->device->ctx = calloc (1, sizeof (gaspi_tcp_ctx));
  if (NULL == gctx->device->ctx)
  {
    free (gctx->device);
    return -1;
  }

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  struct tcp_dev_args *dev_args = malloc (sizeof (struct tcp_dev_args));

  if (NULL == dev_args)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return -1;
  }

  dev_args->peers_num = gctx->tnc;
  dev_args->id = gctx->rank;
  dev_args->port =
    gctx->config->dev_config.params.tcp.port + gctx->local_rank;

  tcp_dev_ctx->device_channel = tcp_dev_init_device (dev_args);

  if (tcp_dev_ctx->device_channel < 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to initialize device.");
    return -1;
  }

  /* user did not choose so we set the network type */
  if (!gctx->config->user_net)
  {
    gctx->config->network = GASPI_ETHERNET;
  }

  if (gctx->config->net_info)
  {
    pgaspi_tcp_dev_print_info (gctx);
  }

  /* Passive channel (SRQ) */
  tcp_dev_ctx->srqP =
    gaspi_sn_connect2port ("localhost",
                           gctx->config->dev_config.params.tcp.port +
                           gctx->local_rank, CONN_TIMEOUT);
  if (tcp_dev_ctx->srqP == -1)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create passive channel connection");
    return -1;
  }

  tcp_dev_ctx->channelP = tcp_dev_create_passive_channel();
  if (tcp_dev_ctx->channelP == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create passive channel.");
    return -1;
  }

  /* Completion Queues */
  tcp_dev_ctx->scqGroups =
    tcp_dev_create_cq (gctx->config->queue_size_max, NULL);
  if (tcp_dev_ctx->scqGroups == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create groups send completion queue.");
    return -1;
  }

  tcp_dev_ctx->rcqGroups =
    tcp_dev_create_cq (gctx->config->queue_size_max, NULL);
  if (tcp_dev_ctx->rcqGroups == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to create groups receive completion queue.");
    return -1;
  }

  tcp_dev_ctx->scqP = tcp_dev_create_cq (gctx->config->queue_size_max, NULL);
  if (tcp_dev_ctx->scqP == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to create passive send completion queue.");
    return -1;
  }

  tcp_dev_ctx->rcqP =
    tcp_dev_create_cq (gctx->config->queue_size_max, tcp_dev_ctx->channelP);
  if (tcp_dev_ctx->rcqP == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to create passive recv completion queue.");
    return -1;
  }

  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    tcp_dev_ctx->scqC[c] =
      tcp_dev_create_cq (gctx->config->queue_size_max, NULL);
    if (tcp_dev_ctx->scqC[c] == NULL)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create IO completion queue.");
      return -1;
    }
  }

  /* Queues (QPs) */
  tcp_dev_ctx->qpGroups = tcp_dev_create_queue (tcp_dev_ctx->scqGroups,
                                                tcp_dev_ctx->rcqGroups);
  if (tcp_dev_ctx->qpGroups == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create queue for groups.");
    return -1;
  }

  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    tcp_dev_ctx->qpC[c] = tcp_dev_create_queue (tcp_dev_ctx->scqC[c], NULL);
    if (tcp_dev_ctx->qpC[c] == NULL)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create queue %d for IO.", c);
      return -1;
    }
  }

  tcp_dev_ctx->qpP = tcp_dev_create_queue (tcp_dev_ctx->scqP,
                                           tcp_dev_ctx->rcqP);
  if (tcp_dev_ctx->qpP == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create queue for passive.");
    return -1;
  }

  gaspi_tcp_dev_status_t _dev_status = gaspi_tcp_dev_status_get();

  while (GASPI_TCP_DEV_STATUS_DOWN == _dev_status)
  {
    GASPI_DELAY();
    _dev_status = gaspi_tcp_dev_status_get();
  }

  if (GASPI_TCP_DEV_STATUS_FAILED == _dev_status)
  {
    return -1;
  }

  free (dev_args);

  return 0;
}

int
pgaspi_dev_cleanup_core (gaspi_context_t * const gctx)
{
  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  if (tcp_dev_stop_device (tcp_dev_ctx->device_channel) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to stop device.");
  }

  /* Destroy posting queues and associated channels */
  tcp_dev_destroy_queue (tcp_dev_ctx->qpGroups);
  tcp_dev_destroy_queue (tcp_dev_ctx->qpP);

  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    tcp_dev_destroy_queue (tcp_dev_ctx->qpC[c]);
  }

  if (tcp_dev_ctx->srqP)
  {
    if (close (tcp_dev_ctx->srqP) < 0)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to close srqP.");
    }
  }

  if (tcp_dev_ctx->channelP)
  {
    tcp_dev_destroy_passive_channel (tcp_dev_ctx->channelP);
  }

  /* Now we can destroy the resources for completion and incoming data */
  tcp_dev_destroy_cq (tcp_dev_ctx->scqGroups);
  tcp_dev_destroy_cq (tcp_dev_ctx->rcqGroups);
  tcp_dev_destroy_cq (tcp_dev_ctx->scqP);
  tcp_dev_destroy_cq (tcp_dev_ctx->rcqP);

  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    tcp_dev_destroy_cq (tcp_dev_ctx->scqC[c]);
  }
  close (tcp_dev_ctx->device_channel);

  free (gctx->device->ctx);
  gctx->device->ctx = NULL;
  free (gctx->device);
  gctx->device = NULL;

  return 0;
}
