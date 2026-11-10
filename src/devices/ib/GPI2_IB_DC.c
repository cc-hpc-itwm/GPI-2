/*
Copyright (c) Fraunhofer ITWM, 2013-2026

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

/* Dynamically Connected (DC) transport of the IB device (GASPI_IB_DCT).

   Instead of an RC queue pair per channel and peer, each channel (groups,
   passive, each communication queue) has DC initiators (DCIs), which reach
   any peer, and one DC target (DCT), which any peer reaches. The
   destination (address handle of the peer and its DCT number) is given in
   every work request. The DCT numbers travel in struct ib_ctx_info (qpn*)
   like the RC QP numbers.

   A DCI reconnects (about 2 us) whenever its target changes, so groups and
   each queue have a pool of DCIs, rank r on DCI r % pool; passive has one.
   Each queue has further DCIs for the parts of large reads. DCIs are
   created on first use. */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include "GASPI_types.h"
#include "GPI2.h"
#include "GPI2_IB.h"
#include "GPI2_Sys.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

/* Bound on waiting for the flushed completions behind a failed request */
#define GPI2_IB_DC_RECOVER_MS (60000)

/* Errors that make GASPI_IB_DCT unusable are reported also in the
   production library: DC is an explicit request of the user. */
#define GPI2_IB_DC_ERROR(rc, what)                                      \
  fprintf (stderr, "GASPI_IB_DCT: %s failed: %s (see dmesg)\n", what,   \
           strerror (rc))

static int
_pgaspi_ib_dc_path_mtu (gaspi_context_t const *const gctx,
                        enum ibv_mtu *const mtu)
{
  switch (gctx->config->dev_config.params.ib.mtu)
  {
    case 1024:
      *mtu = IBV_MTU_1024;
      return 0;
    case 2048:
      *mtu = IBV_MTU_2048;
      return 0;
    case 4096:
      *mtu = IBV_MTU_4096;
      return 0;
    default:
      fprintf (stderr, "GASPI_IB_DCT: invalid MTU in configuration (%d)\n",
               gctx->config->dev_config.params.ib.mtu);
      return -1;
  }
}

/* Address vector of a port: the local one for the DC QPs themselves, a
   peer's one for its address handle. */
static void
_pgaspi_ib_dc_av (gaspi_context_t const *const gctx,
                  struct ibv_ah_attr *const av,
                  const int lid, union ibv_gid const *const gid)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  memset (av, 0, sizeof (*av));
  av->port_num = ib_dev_ctx->ib_port;

  if (gctx->config->network == GASPI_IB)
  {
    av->is_global = 0;
    av->dlid = (uint16_t) lid;
  }
  else
  {
    av->is_global = 1;
    av->grh.dgid = *gid;
    av->grh.sgid_index = GASPI_GID_INDEX;
    av->grh.hop_limit = 1;
  }
}

static void
_pgaspi_ib_dc_local_av (gaspi_context_t const *const gctx,
                        struct ibv_ah_attr *const av)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  _pgaspi_ib_dc_av (gctx, av,
                    ib_dev_ctx->port_attr[ib_dev_ctx->ib_port - 1].lid,
                    &ib_dev_ctx->gid);
}

/* A DCT gets its number only in RTR: it must be ready before its number is
   handed out (struct ib_ctx_info). */
static struct ibv_qp *
_pgaspi_ib_dc_create_dct (gaspi_context_t const *const gctx,
                          struct ibv_cq *const cq, struct ibv_srq *const srq)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_qp_init_attr_ex attr;

  memset (&attr, 0, sizeof (attr));
  attr.qp_type = IBV_QPT_DRIVER;
  attr.send_cq = cq;
  attr.recv_cq = cq;
  attr.srq = srq;
  attr.pd = ib_dev_ctx->pd;
  attr.comp_mask = IBV_QP_INIT_ATTR_PD;

  struct mlx5dv_qp_init_attr dv_attr;

  memset (&dv_attr, 0, sizeof (dv_attr));
  dv_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  dv_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
  dv_attr.dc_init_attr.dct_access_key = 0;

  struct ibv_qp *const dct =
    mlx5dv_create_qp (ib_dev_ctx->context, &attr, &dv_attr);
  if (NULL == dct)
  {
    GPI2_IB_DC_ERROR (errno, "creating a DCT");
    return NULL;
  }

  struct ibv_qp_attr qp_attr;

  memset (&qp_attr, 0, sizeof (qp_attr));
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = ib_dev_ctx->ib_port;
  qp_attr.qp_access_flags =
    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
    IBV_ACCESS_REMOTE_ATOMIC;

  int rc = ibv_modify_qp (dct, &qp_attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT
                          | IBV_QP_ACCESS_FLAGS);
  if (rc != 0)
  {
    GPI2_IB_DC_ERROR (rc, "DCT to INIT");
    ibv_destroy_qp (dct);
    return NULL;
  }

  memset (&qp_attr, 0, sizeof (qp_attr));
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.min_rnr_timer = 12;
  _pgaspi_ib_dc_local_av (gctx, &qp_attr.ah_attr);
  if (_pgaspi_ib_dc_path_mtu (gctx, &qp_attr.path_mtu) != 0)
  {
    ibv_destroy_qp (dct);
    return NULL;
  }

  rc = ibv_modify_qp (dct, &qp_attr,
                      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_MIN_RNR_TIMER
                      | IBV_QP_AV);
  if (rc != 0)
  {
    GPI2_IB_DC_ERROR (rc, "DCT to RTR");
    ibv_destroy_qp (dct);
    return NULL;
  }

  return dct;
}

/* A DCI needs no attribute of a peer: it goes from RESET to RTS at once. */
static int
_pgaspi_ib_dc_dci_ready (gaspi_context_t const *const gctx,
                         struct ibv_qp *const qp)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_qp_attr qp_attr;

  memset (&qp_attr, 0, sizeof (qp_attr));
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = ib_dev_ctx->ib_port;

  int rc = ibv_modify_qp (qp, &qp_attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT);
  if (rc != 0)
  {
    GPI2_IB_DC_ERROR (rc, "DCI to INIT");
    return -1;
  }

  memset (&qp_attr, 0, sizeof (qp_attr));
  qp_attr.qp_state = IBV_QPS_RTR;
  _pgaspi_ib_dc_local_av (gctx, &qp_attr.ah_attr);
  if (_pgaspi_ib_dc_path_mtu (gctx, &qp_attr.path_mtu) != 0)
  {
    return -1;
  }

  rc = ibv_modify_qp (qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV);
  if (rc != 0)
  {
    GPI2_IB_DC_ERROR (rc, "DCI to RTR");
    return -1;
  }

  memset (&qp_attr, 0, sizeof (qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.timeout = GASPI_QP_TIMEOUT;
  qp_attr.retry_cnt = GASPI_QP_RETRY;
  qp_attr.rnr_retry = GASPI_QP_RETRY;
  qp_attr.sq_psn = 0;
  qp_attr.max_rd_atomic = ib_dev_ctx->device_attr.max_qp_rd_atom;

  rc = ibv_modify_qp (qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT
                      | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN
                      | IBV_QP_MAX_QP_RD_ATOMIC);
  if (rc != 0)
  {
    GPI2_IB_DC_ERROR (rc, "DCI to RTS");
    return -1;
  }

  return 0;
}

static int
_pgaspi_ib_dc_create_dci (gaspi_context_t const *const gctx,
                          struct ibv_cq *const cq, const uint32_t depth,
                          struct pgaspi_ib_dci *const dci)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_qp_init_attr_ex attr;

  memset (&attr, 0, sizeof (attr));
  attr.qp_type = IBV_QPT_DRIVER;
  attr.send_cq = cq;
  attr.recv_cq = cq;
  attr.pd = ib_dev_ctx->pd;
  attr.cap.max_send_wr = depth;
  attr.cap.max_send_sge = 1;
  attr.cap.max_inline_data = MAX_INLINE_BYTES;
  attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  attr.send_ops_flags =
    IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM
    | IBV_QP_EX_WITH_RDMA_READ | IBV_QP_EX_WITH_SEND
    | IBV_QP_EX_WITH_SEND_WITH_IMM | IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD
    | IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP;

  struct mlx5dv_qp_init_attr dv_attr;

  memset (&dv_attr, 0, sizeof (dv_attr));
  dv_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  dv_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;

  dci->qp = mlx5dv_create_qp (ib_dev_ctx->context, &attr, &dv_attr);
  if (NULL == dci->qp)
  {
    GPI2_IB_DC_ERROR (errno, "creating a DCI");
    return -1;
  }

  dci->qpx = ibv_qp_to_qp_ex (dci->qp);
  dci->mqpx = mlx5dv_qp_ex_from_ibv_qp_ex (dci->qpx);

  if (_pgaspi_ib_dc_dci_ready (gctx, dci->qp) != 0)
  {
    ibv_destroy_qp (dci->qp);
    memset (dci, 0, sizeof (*dci));
    return -1;
  }

  return 0;
}

static int
_pgaspi_ib_dc_destroy_dci (struct pgaspi_ib_dci *const dci)
{
  if (dci->qp != NULL && ibv_destroy_qp (dci->qp))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy DCI (libibverbs)");
    return -1;
  }

  memset (dci, 0, sizeof (*dci));

  return 0;
}

static int
_pgaspi_ib_dc_destroy_channel (struct pgaspi_ib_dc_channel *const chan)
{
  int rc = 0;

  for (int i = 0; i < chan->num; i++)
  {
    rc |= _pgaspi_ib_dc_destroy_dci (&chan->pool[i]);
  }

  for (int i = 0; i < chan->num_read; i++)
  {
    rc |= _pgaspi_ib_dc_destroy_dci (&chan->read[i]);
  }

  chan->num = 0;
  chan->num_read = 0;

  return rc;
}

static void
_pgaspi_ib_dc_init_channel (struct ibv_cq *const cq, const uint32_t depth,
                            const int num, const int num_read,
                            const uint8_t * const slot,
                            struct pgaspi_ib_dc_channel *const chan)
{
  chan->cq = cq;
  chan->depth = depth;
  chan->slot = slot;
  chan->num_read = num_read;
  chan->num = num;
}

/* Full depth: one rank may get all requests of the channel */
int
pgaspi_ib_dc_open_dci (struct pgaspi_ib_dc_channel *const chan,
                       struct pgaspi_ib_dci *const dci)
{
  int rc = 0;

  lock_gaspi (&chan->lock);

  if (NULL == dci->qp)
  {
    struct pgaspi_ib_dci new_dci;

    rc = _pgaspi_ib_dc_create_dci (&glb_gaspi_ctx, chan->cq, chan->depth,
                                   &new_dci);
    if (0 == rc)
    {
      dci->qpx = new_dci.qpx;
      dci->mqpx = new_dci.mqpx;
      __atomic_store_n (&dci->qp, new_dci.qp, __ATOMIC_RELEASE);
    }
  }

  unlock_gaspi (&chan->lock);

  return rc;
}

static int
_pgaspi_ib_dc_destroy_dct (struct ibv_qp **const dct)
{
  if (*dct != NULL && ibv_destroy_qp (*dct))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy DCT (libibverbs)");
    return -1;
  }

  *dct = NULL;

  return 0;
}

int
pgaspi_ib_dc_queue_create (gaspi_context_t const *const gctx,
                           const unsigned int id)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  ib_dev_ctx->dctC[id] =
    _pgaspi_ib_dc_create_dct (gctx, ib_dev_ctx->scqC[id], ib_dev_ctx->srqC);
  if (NULL == ib_dev_ctx->dctC[id])
  {
    return -1;
  }

  _pgaspi_ib_dc_init_channel (ib_dev_ctx->scqC[id],
                              gctx->config->queue_size_max,
                              ib_dev_ctx->dc_pool, GPI2_IB_DC_READ_PARTS - 1,
                              ib_dev_ctx->dc_slot, &ib_dev_ctx->dciC[id]);
  ib_dev_ctx->dc_parts[id] = 0;

  for (gaspi_rank_t i = 0; i < gctx->tnc; i++)
  {
    ib_dev_ctx->local_info[i].qpnC[id] = ib_dev_ctx->dctC[id]->qp_num;
  }

  return 0;
}

int
pgaspi_ib_dc_queue_delete (gaspi_context_t const *const gctx,
                           const unsigned int id)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (_pgaspi_ib_dc_destroy_channel (&ib_dev_ctx->dciC[id]) != 0
      || _pgaspi_ib_dc_destroy_dct (&ib_dev_ctx->dctC[id]) != 0)
  {
    return -1;
  }

  for (gaspi_rank_t i = 0; i < gctx->tnc; i++)
  {
    ib_dev_ctx->local_info[i].qpnC[id] = 0;
    ib_dev_ctx->remote_info[i].qpnC[id] = 0;
  }

  return 0;
}

/* Called at the end of pgaspi_dev_init_core, with the CQs, srqP and the
   local info in place. */
int
pgaspi_ib_dc_init (gaspi_context_t const *const gctx)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (!mlx5dv_is_supported (ib_dev_ctx->ib_dev))
  {
    fprintf (stderr, "GASPI_IB_DCT: device %s does not support mlx5dv\n",
             ibv_get_device_name (ib_dev_ctx->ib_dev));
    return -1;
  }

  ib_dev_ctx->dc_pool = MIN (GPI2_IB_DC_POOL, (int) gctx->tnc);

  ib_dev_ctx->dc_slot = (uint8_t *) malloc (gctx->tnc);
  ib_dev_ctx->dc_slot0 = (uint8_t *) calloc (gctx->tnc, 1);
  if (NULL == ib_dev_ctx->dc_slot || NULL == ib_dev_ctx->dc_slot0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return -1;
  }

  for (gaspi_rank_t i = 0; i < gctx->tnc; i++)
  {
    ib_dev_ctx->dc_slot[i] = (uint8_t) (i % ib_dev_ctx->dc_pool);
  }

  /* Receives for the sync flags of all peers (pgaspi_ib_post_group_recvs),
     as many as RC posts per peer */
  const int grp_recvs = MIN (GPI2_IB_GRP_RECV_DEPTH * gctx->tnc,
                             ib_dev_ctx->device_attr.max_srq_wr);

  struct ibv_srq_init_attr srq_attr;

  memset (&srq_attr, 0, sizeof (srq_attr));
  srq_attr.attr.max_wr = grp_recvs;
  srq_attr.attr.max_sge = 1;

  ib_dev_ctx->srqGroups = ibv_create_srq (ib_dev_ctx->pd, &srq_attr);
  if (NULL == ib_dev_ctx->srqGroups)
  {
    GPI2_IB_DC_ERROR (errno, "creating the groups SRQ");
    return -1;
  }

  if (pgaspi_ib_post_group_recvs (gctx, 0, grp_recvs) != 0)
  {
    return -1;
  }

  srq_attr.attr.max_wr = 1;
  ib_dev_ctx->srqC = ibv_create_srq (ib_dev_ctx->pd, &srq_attr);
  if (NULL == ib_dev_ctx->srqC)
  {
    GPI2_IB_DC_ERROR (errno, "creating the queues SRQ");
    return -1;
  }

  ib_dev_ctx->ah =
    (struct ibv_ah **) calloc (gctx->tnc, sizeof (struct ibv_ah *));
  if (NULL == ib_dev_ctx->ah)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return -1;
  }

  /* Groups */
  ib_dev_ctx->dctGroups =
    _pgaspi_ib_dc_create_dct (gctx, ib_dev_ctx->rcqGroups,
                              ib_dev_ctx->srqGroups);
  if (NULL == ib_dev_ctx->dctGroups)
  {
    return -1;
  }

  _pgaspi_ib_dc_init_channel (ib_dev_ctx->scqGroups,
                              gctx->config->queue_size_max,
                              ib_dev_ctx->dc_pool, 0, ib_dev_ctx->dc_slot,
                              &ib_dev_ctx->dciGroups);

  /* Passive */
  ib_dev_ctx->dctP =
    _pgaspi_ib_dc_create_dct (gctx, ib_dev_ctx->rcqP, ib_dev_ctx->srqP);
  if (NULL == ib_dev_ctx->dctP)
  {
    return -1;
  }

  _pgaspi_ib_dc_init_channel (ib_dev_ctx->scqP,
                              gctx->config->passive_queue_size_max, 1, 0,
                              ib_dev_ctx->dc_slot0, &ib_dev_ctx->dciP);

  /* Communication queues */
  for (unsigned int c = 0; c < gctx->config->queue_num; c++)
  {
    if (pgaspi_ib_dc_queue_create (gctx, c) != 0)
    {
      return -1;
    }
  }

  for (gaspi_rank_t i = 0; i < gctx->tnc; i++)
  {
    ib_dev_ctx->local_info[i].qpnGroup = ib_dev_ctx->dctGroups->qp_num;
    ib_dev_ctx->local_info[i].qpnP = ib_dev_ctx->dctP->qp_num;
    ib_dev_ctx->local_info[i].dc = 1;
  }

  return 0;
}

int
pgaspi_ib_dc_connect (gaspi_context_t const *const gctx, const int i)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ib_dev_ctx->ah[i] != NULL)
  {
    return 0;
  }

  struct ibv_ah_attr av;

  _pgaspi_ib_dc_av (gctx, &av, ib_dev_ctx->remote_info[i].lid,
                    &ib_dev_ctx->remote_info[i].gid);

  ib_dev_ctx->ah[i] = ibv_create_ah (ib_dev_ctx->pd, &av);
  if (NULL == ib_dev_ctx->ah[i])
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create address handle for %d", i);
    return -1;
  }

  return 0;
}

int
pgaspi_ib_dc_disconnect (gaspi_context_t const *const gctx, const int i)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ib_dev_ctx->ah[i] != NULL && ibv_destroy_ah (ib_dev_ctx->ah[i]))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy address handle (libibverbs)");
    return -1;
  }

  ib_dev_ctx->ah[i] = NULL;

  /* The shared DCIs and DCTs stay: the local info (DCT numbers) remains
     valid for a later connection. */
  memset (&ib_dev_ctx->remote_info[i], 0, sizeof (struct ib_ctx_info));

  return 0;
}

int
pgaspi_ib_dc_cleanup (gaspi_context_t const *const gctx)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ib_dev_ctx->ah != NULL)
  {
    for (gaspi_rank_t i = 0; i < gctx->tnc; i++)
    {
      if (pgaspi_ib_dc_disconnect (gctx, i) != 0)
      {
        return -1;
      }
    }

    free (ib_dev_ctx->ah);
    ib_dev_ctx->ah = NULL;
  }

  free (ib_dev_ctx->dc_slot);
  ib_dev_ctx->dc_slot = NULL;
  free (ib_dev_ctx->dc_slot0);
  ib_dev_ctx->dc_slot0 = NULL;

  for (unsigned int c = 0; c < GASPI_MAX_QP; c++)
  {
    if (_pgaspi_ib_dc_destroy_channel (&ib_dev_ctx->dciC[c]) != 0
        || _pgaspi_ib_dc_destroy_dct (&ib_dev_ctx->dctC[c]) != 0)
    {
      return -1;
    }
  }

  if (_pgaspi_ib_dc_destroy_channel (&ib_dev_ctx->dciGroups) != 0
      || _pgaspi_ib_dc_destroy_dct (&ib_dev_ctx->dctGroups) != 0
      || _pgaspi_ib_dc_destroy_channel (&ib_dev_ctx->dciP) != 0
      || _pgaspi_ib_dc_destroy_dct (&ib_dev_ctx->dctP) != 0)
  {
    return -1;
  }

  if (ib_dev_ctx->srqGroups != NULL && ibv_destroy_srq (ib_dev_ctx->srqGroups))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy SRQ (libibverbs)");
    return -1;
  }
  ib_dev_ctx->srqGroups = NULL;

  if (ib_dev_ctx->srqC != NULL && ibv_destroy_srq (ib_dev_ctx->srqC))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to destroy SRQ (libibverbs)");
    return -1;
  }
  ib_dev_ctx->srqC = NULL;

  return 0;
}

static int
_pgaspi_ib_dc_reset_dci (gaspi_context_t const *const gctx,
                         struct pgaspi_ib_dci *const dci)
{
  struct ibv_qp_attr qp_attr;
  struct ibv_qp_init_attr init_attr;

  if (NULL == dci->qp)
  {
    return 0;
  }

  int rc = ibv_query_qp (dci->qp, &qp_attr, IBV_QP_STATE, &init_attr);
  if (rc != 0)
  {
    GPI2_IB_DC_ERROR (rc, "DCI state query");
    return -1;
  }

  if (qp_attr.qp_state == IBV_QPS_RTS)
  {
    return 0;
  }

  memset (&qp_attr, 0, sizeof (qp_attr));
  qp_attr.qp_state = IBV_QPS_RESET;

  rc = ibv_modify_qp (dci->qp, &qp_attr, IBV_QP_STATE);
  if (rc != 0)
  {
    GPI2_IB_DC_ERROR (rc, "DCI to RESET");
    return -1;
  }

  return _pgaspi_ib_dc_dci_ready (gctx, dci->qp);
}

/* A failed request moves its DCI to the error state and flushes the
   requests behind it, also to other peers. Poll the outstanding completions
   and bring the failed DCIs back to RTS. */
int
pgaspi_ib_dc_recover (gaspi_context_t const *const gctx,
                      struct pgaspi_ib_dc_channel *const chan,
                      struct ibv_cq *const cq, gaspi_uint * const outstanding)
{
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  while (__atomic_load_n (outstanding, __ATOMIC_RELAXED) > 0)
  {
    struct ibv_wc wc;
    const int ne = ibv_poll_cq (cq, 1, &wc);

    if (ne < 0)
    {
      return -1;
    }

    if (ne > 0)
    {
      __atomic_sub_fetch (outstanding, 1, __ATOMIC_RELAXED);
      continue;
    }

    const float ms =
      (float) (gaspi_get_cycles () - s0) * gctx->cycles_to_msecs;
    if (ms > GPI2_IB_DC_RECOVER_MS)
    {
      fprintf (stderr, "GASPI_IB_DCT: %u completions missing after an error\n",
               __atomic_load_n (outstanding, __ATOMIC_RELAXED));
      return -1;
    }
  }

  int rc = 0;

  for (int i = 0; i < chan->num; i++)
  {
    rc |= _pgaspi_ib_dc_reset_dci (gctx, &chan->pool[i]);
  }

  for (int i = 0; i < chan->num_read; i++)
  {
    rc |= _pgaspi_ib_dc_reset_dci (gctx, &chan->read[i]);
  }

  return rc;
}

/* Post a chain of send requests, built as for RC, on a DCI to the DCT dctn
   of rank. All or none of the chain is posted. */
int
pgaspi_ib_dc_post (gaspi_ib_ctx * const ib_dev_ctx,
                   struct pgaspi_ib_dci *const dci,
                   const gaspi_rank_t rank, const uint32_t dctn,
                   struct ibv_send_wr *wr)
{
  struct ibv_qp_ex *const qpx = dci->qpx;
  struct ibv_ah *const ah = ib_dev_ctx->ah[rank];

  if (NULL == ah)
  {
    return EINVAL;
  }

  ibv_wr_start (qpx);

  for (; wr != NULL; wr = wr->next)
  {
    qpx->wr_id = wr->wr_id;
    qpx->wr_flags = wr->send_flags;

    switch (wr->opcode)
    {
      case IBV_WR_RDMA_WRITE:
        ibv_wr_rdma_write (qpx, wr->wr.rdma.rkey, wr->wr.rdma.remote_addr);
        break;
      case IBV_WR_RDMA_WRITE_WITH_IMM:
        ibv_wr_rdma_write_imm (qpx, wr->wr.rdma.rkey,
                               wr->wr.rdma.remote_addr, wr->imm_data);
        break;
      case IBV_WR_RDMA_READ:
        ibv_wr_rdma_read (qpx, wr->wr.rdma.rkey, wr->wr.rdma.remote_addr);
        break;
      case IBV_WR_SEND:
        ibv_wr_send (qpx);
        break;
      case IBV_WR_SEND_WITH_IMM:
        ibv_wr_send_imm (qpx, wr->imm_data);
        break;
      case IBV_WR_ATOMIC_FETCH_AND_ADD:
        ibv_wr_atomic_fetch_add (qpx, wr->wr.atomic.rkey,
                                 wr->wr.atomic.remote_addr,
                                 wr->wr.atomic.compare_add);
        break;
      case IBV_WR_ATOMIC_CMP_AND_SWP:
        ibv_wr_atomic_cmp_swp (qpx, wr->wr.atomic.rkey,
                               wr->wr.atomic.remote_addr,
                               wr->wr.atomic.compare_add,
                               wr->wr.atomic.swap);
        break;
      default:
        ibv_wr_abort (qpx);
        return EINVAL;
    }

    mlx5dv_wr_set_dc_addr (dci->mqpx, ah, dctn, 0);

    if ((wr->send_flags & IBV_SEND_INLINE) && wr->num_sge == 1)
    {
      ibv_wr_set_inline_data (qpx, (void *) (uintptr_t) wr->sg_list[0].addr,
                              wr->sg_list[0].length);
    }
    else
    {
      ibv_wr_set_sge_list (qpx, wr->num_sge, wr->sg_list);
    }
  }

  return ibv_wr_complete (qpx);
}
