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

#include <errno.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <unistd.h>

#ifdef GPI2_EXP_VERBS
#include <math.h>
#endif

#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_IB.h"

/* Globals */
static const char *port_state_str[] = {
  "NOP",
  "Down",
  "Initializing",
  "Armed",
  "Active",
  "Active deferred"
};

static const char *port_phy_state_str[] = {
  "No state change",
  "Sleep",
  "Polling",
  "Disabled",
  "PortConfigurationTraining",
  "LinkUp",
  "LinkErrorRecovery",
  "PhyTest"
};

static const char *
link_layer_str (uint8_t link_layer)
{
  switch (link_layer)
  {
    case IBV_LINK_LAYER_UNSPECIFIED:
    case IBV_LINK_LAYER_INFINIBAND:
      return "IB";
    case IBV_LINK_LAYER_ETHERNET:
      return "Ethernet(RoCE)";
    default:
      return "Unknown";
  }
}

static int
pgaspi_null_gid (union ibv_gid *gid)
{
  return !(gid->raw[8] | gid->raw[9] | gid->raw[10] | gid->raw[11] |
           gid->raw[12] | gid->raw[13] | gid->raw[14] | gid->raw[15]);
}

static void
pgaspi_ib_dev_print_info (gaspi_context_t * const gctx,
                          gaspi_ib_ctx * const ib_dev_ctx, int dev_idx)
{
  char boardIDbuf[256];

  gaspi_printf ("<<<<<<<<<<<<<<<<IB-info>>>>>>>>>>>>>>>>>>>\n");
  gaspi_printf ("\tib_dev     : %d (%s)\n",
                dev_idx, ibv_get_device_name (ib_dev_ctx->dev_list[dev_idx]));
  gaspi_printf ("\tca type    : %d\n", ib_dev_ctx->device_attr.vendor_part_id);

  if (gctx->config->dev_config.params.ib.mtu == 0)
    gaspi_printf ("\tmtu        : (active_mtu)\n");
  else
    gaspi_printf ("\tmtu        : %d (user)\n",
                  gctx->config->dev_config.params.ib.mtu);

  gaspi_printf ("\tfw_version : %s\n", ib_dev_ctx->device_attr.fw_ver);
  gaspi_printf ("\thw_version : %x\n", ib_dev_ctx->device_attr.hw_ver);

  if (ibv_read_sysfs_file
      (ib_dev_ctx->ib_dev->ibdev_path, "board_id", boardIDbuf,
       sizeof (boardIDbuf)) > 0)
  {
    gaspi_printf ("\tpsid       : %s\n", boardIDbuf);
  }

  gaspi_printf ("\t# ports    : %d\n", ib_dev_ctx->device_attr.phys_port_cnt);
  gaspi_printf ("\t# rd_atom  : %d\n", ib_dev_ctx->device_attr.max_qp_rd_atom);

  int id0[2] = { 0, 0 };
  int id1[2] = { 0, 0 };

  for (uint8_t p = 0; p < MIN (ib_dev_ctx->device_attr.phys_port_cnt, GASPI_MAX_PORTS);
       p++)
  {
    gaspi_printf ("\tport Nr    : %d\n", p + 1);

    id0[p] =
      ib_dev_ctx->port_attr[p].state < 6 ? ib_dev_ctx->port_attr[p].state : 0;
    gaspi_printf ("\t  state      : %s\n", port_state_str[id0[p]]);

    id1[p] =
      ib_dev_ctx->port_attr[p].phys_state <
      8 ? ib_dev_ctx->port_attr[p].phys_state : 3;
    gaspi_printf ("\t  phy state  : %s\n", port_phy_state_str[id1[p]]);

    gaspi_printf ("\t  link layer : %s\n",
                  link_layer_str (ib_dev_ctx->port_attr[p].link_layer));
  }

  gaspi_printf ("\tusing port : %d\n", ib_dev_ctx->ib_port);
  gaspi_printf ("\tmtu        : %d\n", gctx->config->dev_config.params.ib.mtu);

  if (gctx->config->network == GASPI_ROCE)
  {
    if (!pgaspi_null_gid (&ib_dev_ctx->gid))
    {
      gaspi_printf
        ("gid[0]: %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x\n",
         ib_dev_ctx->gid.raw[0], ib_dev_ctx->gid.raw[1],
         ib_dev_ctx->gid.raw[2], ib_dev_ctx->gid.raw[3],
         ib_dev_ctx->gid.raw[4], ib_dev_ctx->gid.raw[5],
         ib_dev_ctx->gid.raw[6], ib_dev_ctx->gid.raw[7],
         ib_dev_ctx->gid.raw[8], ib_dev_ctx->gid.raw[9],
         ib_dev_ctx->gid.raw[10], ib_dev_ctx->gid.raw[11],
         ib_dev_ctx->gid.raw[12], ib_dev_ctx->gid.raw[13],
         ib_dev_ctx->gid.raw[14], ib_dev_ctx->gid.raw[15]);
    }
  }
}

int
pgaspi_dev_query_port(gaspi_ib_ctx *const ib_dev_ctx,
                      char const exit_on_error,
                      gaspi_int *configured_port)
{
  for (uint8_t p = 0;
       p < MIN (ib_dev_ctx->device_attr.phys_port_cnt, GASPI_MAX_PORTS); p++)
  {
    if (ibv_query_port (ib_dev_ctx->context, (p + 1),
                        &ib_dev_ctx->port_attr[p]))
    {
      if (exit_on_error)
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed to query port (%u) (libibverbs)",
                                 (p + 1));
        return -1;
      }
      else
      {
        continue;
      }
    }
    if (ib_dev_ctx->port_attr[p].state != IBV_PORT_ACTIVE)
    {
      if (exit_on_error)
      {
        GASPI_DEBUG_PRINT_ERROR ("No IB active port found.");
        return -1;
      }
      else
      {
        continue;
      }
    }
    if (ib_dev_ctx->port_attr[p].phys_state != PORT_LINK_UP)
    {
      if (exit_on_error)
      {
        GASPI_DEBUG_PRINT_ERROR ("No IB active link found.");
        return -1;
      }
      else
      {
        continue;
      }
    }
    *configured_port = (gaspi_int) p;
    break;
  }

  return 0;
}


int
pgaspi_dev_query_id_dev (gaspi_ib_ctx *ib_dev_ctx,
                         char const exit_on_error,
                         gaspi_int const configured_dev_id)
{
  /* Query device */
  ib_dev_ctx->ib_dev = ib_dev_ctx->dev_list[configured_dev_id];
  if (NULL == ib_dev_ctx->ib_dev)
  {
    if (exit_on_error)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to get device %d (libibverbs)",
                              configured_dev_id);
      return -1;
    }
    else
    {
      return 0;
    }
  }

  if (NULL == ib_dev_ctx->ib_dev)
  {
    if (exit_on_error)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to find IB device.");
      return -1;
    }
    else
    {
      return 0;
    }
  }

  if (ib_dev_ctx->ib_dev->transport_type != IBV_TRANSPORT_IB)
  {
    if (exit_on_error)
    {
      GASPI_DEBUG_PRINT_ERROR ("Device does not support IB transport");
      return -1;
    }
    else
    {
      return 0;
    }
  }

  ib_dev_ctx->context = ibv_open_device (ib_dev_ctx->ib_dev);
  if (NULL == ib_dev_ctx->context)
  {
    if (exit_on_error)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to open IB device (libibverbs)");
      return -1;
    }
    else
    {
      return 0;
    }
  }

  if (ibv_query_device (ib_dev_ctx->context, &ib_dev_ctx->device_attr))
  {
    if (exit_on_error)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to query device (libibverbs)");
      return -1;
    }
    else
    {
      if (ibv_close_device (ib_dev_ctx->context))
      {
        if (exit_on_error)
        {
          GASPI_DEBUG_PRINT_ERROR ("Failed to close device (libibverbs)");
          return -1;
        }
        else
        {
          return 0;
        }
      }
    }
  }

  /* Query port(s) */
  gaspi_int configured_port = -1;
  pgaspi_dev_query_port (ib_dev_ctx, exit_on_error, &configured_port);
  if (configured_port >= 0)
  {
    ib_dev_ctx->ib_port = configured_port + 1;
    return 1;
  }
  else
  {
    return 0;
  }
}

int
pgaspi_dev_init_core(gaspi_context_t *const gctx)
{
  gctx->device = calloc (1, sizeof (gctx->device));
  if (NULL == gctx->device)
  {
    return -1;
  }

  gctx->device->ctx = calloc (1, sizeof (gaspi_ib_ctx));
  if (NULL == gctx->device->ctx)
  {
    free (gctx->device);
    return -1;
  }

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  /* TODO: magic number */
  for (int i = 0; i < 64; i++)
  {
    ib_dev_ctx->wc_grp_send[i].status = IBV_WC_SUCCESS;
  }

  /* Take care of IB device */
  int avail_num_dev = -1;

  ib_dev_ctx->dev_list = ibv_get_device_list (&avail_num_dev);
  if (NULL == ib_dev_ctx->dev_list)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to get device list (libibverbs)");
    return -1;
  }

  gaspi_int configured_dev_id = gctx->config->dev_config.params.ib.netdev_id;
  char exit_on_error;

  /* Has user configured which device to use ? */
  if (configured_dev_id >= 0)
  {
    if (configured_dev_id >= avail_num_dev)
    {
      GASPI_DEBUG_PRINT_ERROR ("Configured netdev_id not available (%d)",
                               configured_dev_id);
      return -1;
    }

    exit_on_error = 1;
    pgaspi_dev_query_id_dev (ib_dev_ctx, exit_on_error, configured_dev_id);

  }
  else
  {
    /* Look for one with IB Transport */
    exit_on_error = 0;
    for (gaspi_int i = 0; i < avail_num_dev; i++)
    {
      if (pgaspi_dev_query_id_dev (ib_dev_ctx, exit_on_error, i))
      {
        configured_dev_id = i;
        break;
      }
    }

    if (configured_dev_id < 0)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to auto-setup device (libibverbs)");
      return -1;
    }
  }

  /* Port check and set */
  if (gctx->config->dev_config.params.ib.port_check)
  {
    /* user didn't set network type, use the one set for the port */
    if (!gctx->config->user_net)
    {
      if (ib_dev_ctx->port_attr[ib_dev_ctx->ib_port - 1].link_layer ==
          IBV_LINK_LAYER_INFINIBAND)
      {
        gctx->config->network = GASPI_IB;
      }

      else if (ib_dev_ctx->port_attr[ib_dev_ctx->ib_port - 1].link_layer ==
               IBV_LINK_LAYER_ETHERNET)
      {
        gctx->config->network = GASPI_ROCE;
      }
    }
    else
    {
      if (gctx->config->network == GASPI_IB)
      {
        if (ib_dev_ctx->port_attr[ib_dev_ctx->ib_port - 1].link_layer !=
            IBV_LINK_LAYER_INFINIBAND)
        {
          GASPI_DEBUG_PRINT_ERROR
                  ("No active Infiniband port with active link found.");
          return -1;
        }
      }
      else if (gctx->config->network == GASPI_ROCE)
      {
        if (ib_dev_ctx->port_attr[ib_dev_ctx->ib_port - 1].link_layer !=
            IBV_LINK_LAYER_ETHERNET)
        {
          GASPI_DEBUG_PRINT_ERROR(
              "No active Ethernet (RoCE) port with active link found.");
          return -1;
        }
      }
    }
  }
  else
  {
    GASPI_PRINT_WARNING ("No port(s) check! Using port 1.");
    ib_dev_ctx->ib_port = 1;
  }

  /* Configure MTU */
  if (gctx->config->network == GASPI_IB)
  {
    if (gctx->config->dev_config.params.ib.mtu == 0)
    {
      switch (ib_dev_ctx->port_attr[ib_dev_ctx->ib_port - 1].active_mtu)
      {
        case IBV_MTU_1024:
          gctx->config->dev_config.params.ib.mtu = 1024;
          break;
        case IBV_MTU_2048:
          gctx->config->dev_config.params.ib.mtu = 2048;
          break;
        case IBV_MTU_4096:
          gctx->config->dev_config.params.ib.mtu = 4096;
          break;
        default:
          break;
      };
    }
  }

  if (gctx->config->network == GASPI_ROCE)
  {
    gctx->config->dev_config.params.ib.mtu = 1024;

    const int ret =
      ibv_query_gid (ib_dev_ctx->context, ib_dev_ctx->ib_port, GASPI_GID_INDEX,
                     &ib_dev_ctx->gid);

    if (ret)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to query gid (RoCE - libiverbs)");
      return -1;
    }
  }

  /*  Print info  */
  if (gctx->config->net_info)
  {
    pgaspi_ib_dev_print_info (gctx, ib_dev_ctx, configured_dev_id);
  }

  ib_dev_ctx->pd = ibv_alloc_pd (ib_dev_ctx->context);
  if (NULL == ib_dev_ctx->pd)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to allocate protection domain (libibverbs)");
    return -1;
  }

  /* Completion channel (for passive communication) */
  ib_dev_ctx->channelP = ibv_create_comp_channel (ib_dev_ctx->context);
  if (NULL == ib_dev_ctx->channelP)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to create completion channel (libibverbs)");
    return -1;
  }

  struct ibv_srq_init_attr srq_attr;

  srq_attr.attr.max_wr = gctx->config->queue_size_max;
  srq_attr.attr.max_sge = 1;

  ib_dev_ctx->srqP = ibv_create_srq (ib_dev_ctx->pd, &srq_attr);
  if (NULL == ib_dev_ctx->srqP)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create SRQ (libibverbs)");
    return -1;
  }

  /* Create default completion queues */
  /* Groups */
#ifdef GPI2_EXP_VERBS
  struct ibv_exp_cq_init_attr cqattr;

  memset (&cqattr, 0, sizeof (cqattr));

  ib_dev_ctx->scqGroups =
    ibv_exp_create_cq (ib_dev_ctx->context, gctx->config->queue_size_max, NULL,
                       NULL, 0, &cqattr);
  if (NULL == ib_dev_ctx->scqGroups)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create CQ (libibverbs)");
    return -1;
  }

  ib_dev_ctx->rcqGroups =
    ibv_exp_create_cq (ib_dev_ctx->context, gctx->config->queue_size_max, NULL,
                       NULL, 0, &cqattr);
  if (NULL == ib_dev_ctx->rcqGroups)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create CQ (libibverbs)");
    return -1;
  }
#else
  ib_dev_ctx->scqGroups =
    ibv_create_cq (ib_dev_ctx->context, gctx->config->queue_size_max, NULL,
                   NULL, 0);
  if (NULL == ib_dev_ctx->scqGroups)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create CQ (libibverbs)");
    return -1;
  }

  ib_dev_ctx->rcqGroups =
    ibv_create_cq (ib_dev_ctx->context, gctx->config->queue_size_max, NULL,
                   NULL, 0);
  if (NULL == ib_dev_ctx->rcqGroups)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create CQ (libibverbs)");
    return -1;
  }
#endif

  /* Passive */
  ib_dev_ctx->scqP =
    ibv_create_cq (ib_dev_ctx->context, gctx->config->passive_queue_size_max,
                   NULL, NULL, 0);
  if (NULL == ib_dev_ctx->scqP)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create CQ (libibverbs)");
    return -1;
  }

  ib_dev_ctx->rcqP =
    ibv_create_cq (ib_dev_ctx->context, gctx->config->queue_size_max, NULL,
                   ib_dev_ctx->channelP, 0);
  if (NULL == ib_dev_ctx->rcqP)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create CQ (libibverbs)");
    return -1;
  }

  if (ibv_req_notify_cq (ib_dev_ctx->rcqP, 0))
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to request CQ notifications (libibverbs)");
    return 1;
  }

  /* One-sided Communication */
  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    ib_dev_ctx->scqC[c] =
      ibv_create_cq (ib_dev_ctx->context, gctx->config->queue_size_max, NULL,
                     NULL, 0);
    if (NULL == ib_dev_ctx->scqC[c])
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create CQ (libibverbs)");
      return -1;
    }
  }

  /* Allocate space for QPs */
  //TODO: could be independent of tnc
  ib_dev_ctx->qpGroups =
    (struct ibv_qp **) calloc (gctx->tnc, sizeof (struct ibv_qp *));
  if (NULL == ib_dev_ctx->qpGroups)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return -1;
  }

  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    ib_dev_ctx->qpC[c] =
      (struct ibv_qp **) calloc (gctx->tnc, sizeof (struct ibv_qp *));
    if (NULL == ib_dev_ctx->qpC[c])
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
      return -1;
    }
  }

  ib_dev_ctx->qpP =
    (struct ibv_qp **) calloc (gctx->tnc, sizeof (struct ibv_qp *));
  if (NULL == ib_dev_ctx->qpP)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return -1;
  }

  /* Zero-fy QP creation state */
  //TODO: could be independent of tnc
  memset (&(ib_dev_ctx->qpC_cstat), 0, GASPI_MAX_QP * sizeof(gaspi_uint));

  ib_dev_ctx->local_info =
    (struct ib_ctx_info *) calloc (gctx->tnc, sizeof (struct ib_ctx_info));
  if (NULL == ib_dev_ctx->local_info)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return -1;
  }

  ib_dev_ctx->remote_info =
    (struct ib_ctx_info *) calloc (gctx->tnc, sizeof (struct ib_ctx_info));
  if (NULL == ib_dev_ctx->remote_info)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return -1;
  }

  for (gaspi_rank_t i = 0; i < gctx->tnc; i++)
  {
    ib_dev_ctx->local_info[i].lid =
      ib_dev_ctx->port_attr[ib_dev_ctx->ib_port - 1].lid;

    struct timeval tv;

    gettimeofday (&tv, NULL);
    srand48 (tv.tv_usec);
    ib_dev_ctx->local_info[i].psn = lrand48() & 0xffffff;

    if (gctx->config->dev_config.params.ib.port_check)
    {
      if (!ib_dev_ctx->local_info[i].lid
          && (gctx->config->network == GASPI_IB))
      {
        GASPI_DEBUG_PRINT_ERROR
          ("Failed to find topology! Is subnet-manager running ?");
        return -1;
      }
    }

    if (gctx->config->network == GASPI_ROCE)
    {
      ib_dev_ctx->local_info[i].gid = ib_dev_ctx->gid;
    }
  }

  return 0;
}

static struct ibv_qp *
_pgaspi_dev_create_qp (gaspi_context_t const *const gctx,
                       struct ibv_cq *send_cq, struct ibv_cq *recv_cq,
                       struct ibv_srq *srq, uint32_t depth)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_qp *qp;

  /* Set initial attributes */
  struct ibv_qp_init_attr qpi_attr;

  memset (&qpi_attr, 0, sizeof (struct ibv_qp_init_attr));
  qpi_attr.cap.max_send_wr = depth;
  qpi_attr.cap.max_recv_wr = depth;
  qpi_attr.cap.max_send_sge = 1;
  qpi_attr.cap.max_recv_sge = 1;
  qpi_attr.cap.max_inline_data = MAX_INLINE_BYTES;
  qpi_attr.qp_type = IBV_QPT_RC;

  qpi_attr.send_cq = send_cq;
  qpi_attr.recv_cq = recv_cq;
  qpi_attr.srq = srq;

  qp = ibv_create_qp (ib_dev_ctx->pd, &qpi_attr);
  if (qp == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create QP (libibverbs)");
    return NULL;
  }

  /* Set to init */
  struct ibv_qp_attr qp_attr;

  memset (&qp_attr, 0, sizeof (struct ibv_qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = ib_dev_ctx->ib_port;
  qp_attr.qp_access_flags =
    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
    IBV_ACCESS_REMOTE_ATOMIC;

  if (ibv_modify_qp (qp, &qp_attr,
                     IBV_QP_STATE
                     | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
  {
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to modify QP (libibverbs)");
    }

    if (ibv_destroy_qp (qp))
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
    }
    return NULL;
  }

  return qp;
}


#ifdef GPI2_EXP_VERBS
static struct ibv_qp *
_pgaspi_dev_create_qp_exp (gaspi_context_t const *const gctx,
                           struct ibv_cq *send_cq, struct ibv_cq *recv_cq,
                           struct ibv_srq *srq, uint32_t depth)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_exp_qp_init_attr attr;
  struct ibv_qp *qp = NULL;
  struct ibv_exp_device_attr dev_attr;

  memset (&attr, 0, sizeof (attr));
  memset (&dev_attr, 0, sizeof (dev_attr));

  attr.pd = ib_dev_ctx->pd;
  attr.cap.max_send_wr = depth;
  attr.cap.max_recv_wr = depth;
  attr.cap.max_send_sge = 1;
  attr.cap.max_inline_data = MAX_INLINE_BYTES;

  attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD | IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
  attr.comp_mask |= IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG;
  attr.max_atomic_arg = pow (2, dev_attr.ext_atom.log_max_atomic_inline);
  attr.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;

  attr.send_cq = send_cq;
  attr.recv_cq = recv_cq;
  attr.srq = srq;

  attr.qp_type = IBV_QPT_RC;

  qp = ibv_exp_create_qp (ib_dev_ctx->context, &attr);
  if (qp == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create QP (libibverbs)");
    return NULL;
  }

  /* Set to init */
  struct ibv_exp_qp_attr qp_attr;

  memset (&qp_attr, 0, sizeof (struct ibv_exp_qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = ib_dev_ctx->ib_port;

  qp_attr.qp_access_flags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
    IBV_ACCESS_REMOTE_ATOMIC;

  struct ibv_exp_qp_attr exp_attr;
  uint64_t exp_flags = 0;

  memset (&exp_attr, 0, sizeof (struct ibv_exp_qp_attr));

  exp_attr.qp_state = qp_attr.qp_state;
  exp_attr.pkey_index = qp_attr.pkey_index;
  exp_attr.port_num = ib_dev_ctx->ib_port;

  exp_attr.qp_access_flags = qp_attr.qp_access_flags;

  exp_flags = IBV_EXP_QP_STATE | IBV_EXP_QP_PKEY_INDEX | IBV_EXP_QP_PORT;
  exp_flags |= IBV_EXP_QP_ACCESS_FLAGS;

  if (ibv_exp_modify_qp (qp, &exp_attr, exp_flags))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to modify QP (libibverbs)");

    if (ibv_destroy_qp (qp))
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
    }
    return NULL;
  }

  return qp;
}
#endif //GPI2_EXP_VERBS

int
pgaspi_dev_comm_queue_delete (gaspi_context_t const *const gctx,
                              const unsigned int id)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  for (int i = 0; i < gctx->tnc; i++)
  {
    if (gctx->ep_conn[i].istat == 0)
    {
      continue;
    }

    if (ib_dev_ctx->qpC[id])
    {
      if (ib_dev_ctx->qpC[id][i])
      {
        if (ibv_destroy_qp (ib_dev_ctx->qpC[id][i]))
        {
          GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
          return -1;
        }
      }
    }

    ib_dev_ctx->remote_info[i].qpnC[id] = 0;
  }

  free (ib_dev_ctx->qpC[id]);

  ib_dev_ctx->qpC[id] = NULL;

  if (1 == ib_dev_ctx->qpC_cstat[id])
  {
    if (ib_dev_ctx->scqC[id])
    {
      if (ibv_destroy_cq (ib_dev_ctx->scqC[id]))
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed to destroy CQ (libibverbs)");
        return -1;
      }

      ib_dev_ctx->scqC[id] = NULL;
    }

    ib_dev_ctx->qpC_cstat[id] = 0;
  }

  return 0;
}

int
pgaspi_dev_comm_queue_create (gaspi_context_t const *const gctx,
                              const unsigned int id,
                              const unsigned short remote_node)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (0 == ib_dev_ctx->qpC_cstat[id])
  {
    /* Completion queue */
    ib_dev_ctx->scqC[id] =
      ibv_create_cq (ib_dev_ctx->context, gctx->config->queue_size_max, NULL,
                     NULL, 0);
    if (!ib_dev_ctx->scqC[id])
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create CQ (libibverbs)");
      return -1;
    }

    /* Queue Pair */
    ib_dev_ctx->qpC[id] =
      (struct ibv_qp **) malloc (gctx->tnc * sizeof (struct ibv_qp *));
    if (ib_dev_ctx->qpC[id] == NULL)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to memory allocation");
      return -1;
    }

    ib_dev_ctx->qpC_cstat[id] = 1;
  }

  if (ib_dev_ctx->qpC[id] == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to memory allocation");
    return -1;
  }

  ib_dev_ctx->qpC[id][remote_node] =
    _pgaspi_dev_create_qp (gctx, ib_dev_ctx->scqC[id], ib_dev_ctx->scqC[id],
                           NULL, gctx->config->queue_size_max);

  if (ib_dev_ctx->qpC[id][remote_node] == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create QP (libibverbs)");
    return -1;
  }

  ib_dev_ctx->local_info[remote_node].qpnC[id] =
    ib_dev_ctx->qpC[id][remote_node]->qp_num;

  return 0;
}

int
pgaspi_dev_comm_queue_is_valid (gaspi_context_t const *const gctx,
                                const unsigned int id)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *)gctx->device->ctx;

  if (ib_dev_ctx->qpC[id] == NULL)
  {
    return GASPI_ERR_INV_QUEUE;
  }

  return 0;
}

int
pgaspi_dev_create_endpoint (gaspi_context_t const *const gctx, const int i,
                            void **info, void **remote_info,
                            size_t * info_size)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ib_dev_ctx->qpGroups[i] == NULL)
  {
    /* Groups QP */
#ifdef GPI2_EXP_VERBS
    ib_dev_ctx->qpGroups[i] =
      _pgaspi_dev_create_qp_exp (gctx, ib_dev_ctx->scqGroups,
                                 ib_dev_ctx->rcqGroups, NULL,
                                 gctx->config->queue_size_max);

#else
    ib_dev_ctx->qpGroups[i] =
      _pgaspi_dev_create_qp (gctx, ib_dev_ctx->scqGroups,
                             ib_dev_ctx->rcqGroups, NULL,
                             gctx->config->queue_size_max);

    if (ib_dev_ctx->qpGroups[i] == NULL)
    {
      return -1;
    }
#endif

    ib_dev_ctx->local_info[i].qpnGroup = ib_dev_ctx->qpGroups[i]->qp_num;

    /* IO QPs */
    for (unsigned int c = 0; c < gctx->config->queue_num; c++)
    {
      ib_dev_ctx->qpC[c][i] =
        _pgaspi_dev_create_qp (gctx, ib_dev_ctx->scqC[c], ib_dev_ctx->scqC[c],
                               NULL, gctx->config->queue_size_max);

      if (ib_dev_ctx->qpC[c][i] == NULL)
        return -1;

      ib_dev_ctx->local_info[i].qpnC[c] = ib_dev_ctx->qpC[c][i]->qp_num;
    }

    /* Passive QP */
    ib_dev_ctx->qpP[i] =
      _pgaspi_dev_create_qp (gctx, ib_dev_ctx->scqP, ib_dev_ctx->rcqP,
                             ib_dev_ctx->srqP,
                             gctx->config->passive_queue_size_max);

    if (ib_dev_ctx->qpP[i] == NULL)
      return -1;

    ib_dev_ctx->local_info[i].qpnP = ib_dev_ctx->qpP[i]->qp_num;
  }

  *info = &ib_dev_ctx->local_info[i];
  *remote_info = &ib_dev_ctx->remote_info[i];
  *info_size = sizeof (struct ib_ctx_info);

  return 0;
}

/* TODO: rename to endpoint */
int
pgaspi_dev_disconnect_context (gaspi_context_t * const gctx, const int i)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ibv_destroy_qp (ib_dev_ctx->qpGroups[i]))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
    return -1;
  }

  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    if (ibv_destroy_qp (ib_dev_ctx->qpC[c][i]))
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
      return -1;
    }
  }

  if (ibv_destroy_qp (ib_dev_ctx->qpP[i]))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
    return -1;
  }

  ib_dev_ctx->local_info[i].qpnGroup = 0;
  ib_dev_ctx->local_info[i].qpnP = 0;

  for (unsigned c = 0; c < gctx->config->queue_num; c++)
  {
    ib_dev_ctx->local_info[i].qpnC[c] = 0;
  }

  return 0;
}

static int
_pgaspi_dev_qp_set_ready (gaspi_context_t const *const gctx, struct ibv_qp *qp,
                          int target, int target_qp)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_qp_attr qp_attr;

  memset (&qp_attr, 0, sizeof (qp_attr));

  switch (gctx->config->dev_config.params.ib.mtu)
  {
    case 1024:
      {
        qp_attr.path_mtu = IBV_MTU_1024;
        break;
      }
    case 2048:
      {
        qp_attr.path_mtu = IBV_MTU_2048;
        break;
      }
    case 4096:
      {
        qp_attr.path_mtu = IBV_MTU_4096;
        break;
      }
    default:
      {
        GASPI_DEBUG_PRINT_ERROR ("Invalid MTU in configuration (%d)",
                                 gctx->config->dev_config.params.ib.mtu);
        return -1;
      }
  };

  /* ready2recv */
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.dest_qp_num = target_qp;
  qp_attr.rq_psn = ib_dev_ctx->remote_info[target].psn;
  qp_attr.max_dest_rd_atomic = ib_dev_ctx->device_attr.max_qp_rd_atom;  //ib_dev_ctx->max_rd_atomic;
  qp_attr.min_rnr_timer = 12;

  if (gctx->config->network == GASPI_IB)
  {
    qp_attr.ah_attr.is_global = 0;
    qp_attr.ah_attr.dlid =
      (unsigned short) ib_dev_ctx->remote_info[target].lid;
  }
  else
  {
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.dgid = ib_dev_ctx->remote_info[target].gid;
    qp_attr.ah_attr.grh.hop_limit = 1;
  }

  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = ib_dev_ctx->ib_port;

  if (ibv_modify_qp (qp, &qp_attr,
                     IBV_QP_STATE
                     | IBV_QP_AV
                     | IBV_QP_PATH_MTU
                     | IBV_QP_DEST_QPN
                     | IBV_QP_RQ_PSN
                     | IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to modify QP (libibverbs)");
    return -1;
  }

  /* ready2send */
  qp_attr.timeout = GASPI_QP_TIMEOUT;
  qp_attr.retry_cnt = GASPI_QP_RETRY;
  qp_attr.rnr_retry = GASPI_QP_RETRY;
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.sq_psn = ib_dev_ctx->local_info[target].psn;
  qp_attr.max_rd_atomic = ib_dev_ctx->device_attr.max_qp_rd_atom;       //ib_dev_ctx->max_rd_atomic;

  if (ibv_modify_qp (qp, &qp_attr,
                     IBV_QP_STATE
                     | IBV_QP_SQ_PSN
                     | IBV_QP_TIMEOUT
                     | IBV_QP_RETRY_CNT
                     | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to modify QP (libibverbs)");
    return -1;
  }

  return 0;
}

int
pgaspi_dev_comm_queue_connect (gaspi_context_t const *const gctx,
                               const unsigned short q, const int i)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  /* Not very nice but we need to wait for info to be available */
  do
  {
    usleep (10);
  }
  while (ib_dev_ctx->remote_info[i].qpnC[q] == 0);

  return _pgaspi_dev_qp_set_ready (gctx,
                                   ib_dev_ctx->qpC[q][i],
                                   i, ib_dev_ctx->remote_info[i].qpnC[q]);
}

/* TODO: rename to endpoint */
int
pgaspi_dev_connect_context (gaspi_context_t const *const gctx, const int i)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (0 != _pgaspi_dev_qp_set_ready (gctx,
                                     ib_dev_ctx->qpGroups[i],
                                     i, ib_dev_ctx->remote_info[i].qpnGroup))
  {
    return -1;
  }

  if (0 != _pgaspi_dev_qp_set_ready (gctx,
                                     ib_dev_ctx->qpP[i],
                                     i, ib_dev_ctx->remote_info[i].qpnP))
  {
    return -1;
  }

  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    if (0 != _pgaspi_dev_qp_set_ready (gctx,
                                       ib_dev_ctx->qpC[c][i],
                                       i, ib_dev_ctx->remote_info[i].qpnC[c]))
    {
      return -1;
    }
  }

  return 0;
}

int
pgaspi_dev_cleanup_core (gaspi_context_t * const gctx)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  for (int i = 0; i < gctx->tnc; i++)
  {
    if (GASPI_ENDPOINT_CREATED == gctx->ep_conn[i].istat)
    {
      if (ibv_destroy_qp (ib_dev_ctx->qpGroups[i]))
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
        return -1;
      }

      if (ibv_destroy_qp (ib_dev_ctx->qpP[i]))
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
        return -1;
      }

      for (unsigned int c = 0; c < gctx->num_queues; c++)
      {
        if (ibv_destroy_qp (ib_dev_ctx->qpC[c][i]))
        {
          GASPI_DEBUG_PRINT_ERROR ("Failed to destroy QP (libibverbs)");
          return -1;
        }
      }
    }
  }

  free (ib_dev_ctx->qpGroups);
  ib_dev_ctx->qpGroups = NULL;

  free (ib_dev_ctx->qpP);
  ib_dev_ctx->qpP = NULL;

  for (unsigned int c = 0; c < gctx->num_queues; c++)
  {
    free (ib_dev_ctx->qpC[c]);
  }

  if (ibv_destroy_cq (ib_dev_ctx->scqGroups))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy CQ (libibverbs)");
    return -1;
  }

  if (ibv_destroy_cq (ib_dev_ctx->rcqGroups))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy CQ (libibverbs)");
    return -1;
  }

  if (ibv_destroy_cq (ib_dev_ctx->scqP))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy CQ (libibverbs)");
    return -1;
  }

  if (ibv_destroy_cq (ib_dev_ctx->rcqP))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy CQ (libibverbs)");
    return -1;
  }

  if (ibv_destroy_srq (ib_dev_ctx->srqP))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy SRQ (libibverbs)");
    return -1;
  }

  if (ibv_dealloc_pd (ib_dev_ctx->pd))
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to de-allocate protection domain (libibverbs)");
    return -1;
  }

  if (ib_dev_ctx->channelP)
  {
    if (ibv_destroy_comp_channel (ib_dev_ctx->channelP))
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to destroy completion channel (libibverbs)");
      return -1;
    }
  }

  if (ibv_close_device (ib_dev_ctx->context))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to close device (libibverbs)");
    return -1;
  }

  if (ib_dev_ctx->dev_list)
  {
    ibv_free_device_list (ib_dev_ctx->dev_list);
  }

  free (ib_dev_ctx->local_info);
  ib_dev_ctx->local_info = NULL;

  free (ib_dev_ctx->remote_info);
  ib_dev_ctx->remote_info = NULL;

  free (gctx->device->ctx);
  gctx->device->ctx = NULL;
  free (gctx->device);
  gctx->device = NULL;

  return 0;
}
