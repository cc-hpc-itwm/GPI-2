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

#ifndef _GPI2_IB_H_
#define _GPI2_IB_H_

#include <infiniband/verbs.h>
#ifdef GPI2_IB_DC
#include <infiniband/mlx5dv.h>
#endif

#include "GPI2_Types.h"

#define GASPI_GID_INDEX   (0)
#define PORT_LINK_UP      (5)
#define MAX_INLINE_BYTES  (128)
#define GASPI_QP_TIMEOUT  (20)
#define GASPI_QP_RETRY    (7)
#define GASPI_MAX_PORTS   (2)

/* IB-specific */
struct ib_ctx_info
{
  int lid;
  int psn;
  union ibv_gid gid;
  /* RC: QP numbers, DC: DCT numbers */
  int qpnGroup;
  int qpnP;
  int qpnC[GASPI_MAX_QP];
  int dc;                       /* transport of the sender (GASPI_IB_DCT) */
};

#ifdef GPI2_IB_DC
/* DC initiator: reaches any DCT, one target at a time */
struct pgaspi_ib_dci
{
  struct ibv_qp *qp;
  struct ibv_qp_ex *qpx;
  struct mlx5dv_qp_ex *mqpx;
};

/* DCIs per channel: a DCI reconnects whenever its target changes */
#define GPI2_IB_DC_POOL (16)

/* gaspi_read from 16 KiB in 2-4 parts: one DC connection reads at ~31 GB/s */
#define GPI2_IB_DC_READ_PARTS (4)
#define GPI2_IB_DC_READ_PART_MIN (16 * 1024)

/* wr_id flag of the further parts: not counted as queue entries */
#define GPI2_IB_DC_PART (1ULL << 63)

/* Rank r always on pool[slot[r]]: keeps the order of requests to r. DCIs
   are created on first use; read[] takes further read parts (queues). */
struct pgaspi_ib_dc_channel
{
  int num;
  int num_read;
  const uint8_t *slot;
  struct ibv_cq *cq;
  uint32_t depth;
  gaspi_lock_t lock;            /* DCI creation */
  struct pgaspi_ib_dci pool[GPI2_IB_DC_POOL];
  struct pgaspi_ib_dci read[GPI2_IB_DC_READ_PARTS - 1];
};
#endif

typedef struct
{
  int ib_port;
  int num_queues;

  struct ibv_device **dev_list;
  struct ibv_device *ib_dev;
  struct ibv_context *context;
  struct ibv_comp_channel *channelP;
  struct ibv_pd *pd;
  struct ibv_device_attr device_attr;
  struct ibv_port_attr port_attr[GASPI_MAX_PORTS];
  struct ibv_cq *scqGroups, *rcqGroups;
  struct ibv_qp **qpGroups;
  struct ibv_wc wc_grp_send[64];
  struct ibv_srq *srqP;
  struct ibv_qp **qpP;
  struct ibv_cq *scqP;
  struct ibv_cq *rcqP;
  struct ibv_cq *scqC[GASPI_MAX_QP];
  struct ibv_qp **qpC[GASPI_MAX_QP];
  union ibv_gid gid;

  struct ib_ctx_info *local_info;
  struct ib_ctx_info *remote_info;

  int qpC_cstat[GASPI_MAX_QP];

  /* The NIC places data with relaxed PCIe ordering: set notifications with
     an atomic, fence notification reads and send the allreduce sync flags as
     immediate data (see GASPI_IB_RELAXED_ORDERING) */
  int relaxed_ordering;

  /* Dynamically Connected transport (GASPI_IB_DCT): per channel a pool of
     initiators (DCIs) and one target (DCT), instead of an RC QP per peer */
  int dc;
#ifdef GPI2_IB_DC
  int dc_pool;                  /* MIN (GPI2_IB_DC_POOL, tnc) */
  uint8_t *dc_slot;             /* [tnc]: rank % dc_pool */
  uint8_t *dc_slot0;            /* [tnc]: 0 (passive channel) */
  gaspi_uint dc_parts[GASPI_MAX_QP];    /* further read parts pending */
  struct pgaspi_ib_dc_channel dciGroups, dciP, dciC[GASPI_MAX_QP];
  struct ibv_qp *dctGroups, *dctP, *dctC[GASPI_MAX_QP];
  struct ibv_srq *srqGroups;    /* sync flags as immediate data */
  struct ibv_srq *srqC;         /* empty, required by the queue DCTs */
  struct ibv_ah **ah;           /* per peer, created at connect */
#endif

} gaspi_ib_ctx;

/* Receive requests posted on each groups QP for the sync flags delivered as
   immediate data by peers with relaxed ordering. A rank receives at most a
   few of them per peer before polling (the collectives synchronize), so a
   small depth suffices; more only delay the sender (RNR retry). */
#define GPI2_IB_GRP_RECV_DEPTH (16)

int pgaspi_ib_post_group_recvs (gaspi_context_t const *const gctx,
                                const int rank, const int count);

#ifdef GPI2_IB_DC
int pgaspi_ib_dc_init (gaspi_context_t const *const gctx);
int pgaspi_ib_dc_queue_create (gaspi_context_t const *const gctx,
                               const unsigned int id);
int pgaspi_ib_dc_queue_delete (gaspi_context_t const *const gctx,
                               const unsigned int id);
int pgaspi_ib_dc_connect (gaspi_context_t const *const gctx, const int i);
int pgaspi_ib_dc_disconnect (gaspi_context_t const *const gctx, const int i);
int pgaspi_ib_dc_cleanup (gaspi_context_t const *const gctx);
int pgaspi_ib_dc_post (gaspi_ib_ctx * const ib_dev_ctx,
                       struct pgaspi_ib_dci *const dci,
                       const gaspi_rank_t rank, const uint32_t dctn,
                       struct ibv_send_wr *wr);
int pgaspi_ib_dc_recover (gaspi_context_t const *const gctx,
                          struct pgaspi_ib_dc_channel *const chan,
                          struct ibv_cq *const cq,
                          gaspi_uint * const outstanding);

int pgaspi_ib_dc_open_dci (struct pgaspi_ib_dc_channel *const chan,
                           struct pgaspi_ib_dci *const dci);

static inline struct pgaspi_ib_dci *
pgaspi_ib_dc_use (struct pgaspi_ib_dc_channel *const chan,
                  struct pgaspi_ib_dci *const dci)
{
  if (__builtin_expect
      (__atomic_load_n (&dci->qp, __ATOMIC_ACQUIRE) == NULL, 0)
      && pgaspi_ib_dc_open_dci (chan, dci) != 0)
  {
    return NULL;
  }

  return dci;
}

static inline struct pgaspi_ib_dci *
pgaspi_ib_dc_dci (struct pgaspi_ib_dc_channel *const chan,
                  const gaspi_rank_t rank)
{
  return pgaspi_ib_dc_use (chan, &chan->pool[chan->slot[rank]]);
}
#endif

/* DC: a queue entry may be a read in parts */
static inline int
pgaspi_ib_queue_cq_size (gaspi_context_t const *const gctx)
{
#ifdef GPI2_IB_DC
  if (((gaspi_ib_ctx *) gctx->device->ctx)->dc)
  {
    return gctx->config->queue_size_max * GPI2_IB_DC_READ_PARTS;
  }
#endif

  return gctx->config->queue_size_max;
}

/* Post a chain of send requests to rank, on a communication queue, on the
   groups channel (collectives, atomics) or on the passive channel. */
static inline int
pgaspi_ib_post_queue (gaspi_ib_ctx * const ib_dev_ctx,
                      const gaspi_queue_id_t queue, const gaspi_rank_t rank,
                      struct ibv_send_wr *const wr)
{
#ifdef GPI2_IB_DC
  if (ib_dev_ctx->dc)
  {
    struct pgaspi_ib_dci *const dci =
      pgaspi_ib_dc_dci (&ib_dev_ctx->dciC[queue], rank);

    if (NULL == dci)
    {
      return ENOMEM;
    }

    return pgaspi_ib_dc_post (ib_dev_ctx, dci, rank,
                              ib_dev_ctx->remote_info[rank].qpnC[queue], wr);
  }
#endif

  struct ibv_send_wr *bad_wr;

  return ibv_post_send (ib_dev_ctx->qpC[queue][rank], wr, &bad_wr);
}

static inline int
pgaspi_ib_post_groups (gaspi_ib_ctx * const ib_dev_ctx,
                       const gaspi_rank_t rank, struct ibv_send_wr *const wr)
{
#ifdef GPI2_IB_DC
  if (ib_dev_ctx->dc)
  {
    struct pgaspi_ib_dci *const dci =
      pgaspi_ib_dc_dci (&ib_dev_ctx->dciGroups, rank);

    if (NULL == dci)
    {
      return ENOMEM;
    }

    return pgaspi_ib_dc_post (ib_dev_ctx, dci, rank,
                              ib_dev_ctx->remote_info[rank].qpnGroup, wr);
  }
#endif

  struct ibv_send_wr *bad_wr;

  return ibv_post_send (ib_dev_ctx->qpGroups[rank], wr, &bad_wr);
}

static inline int
pgaspi_ib_post_passive (gaspi_ib_ctx * const ib_dev_ctx,
                        const gaspi_rank_t rank, struct ibv_send_wr *const wr)
{
#ifdef GPI2_IB_DC
  if (ib_dev_ctx->dc)
  {
    struct pgaspi_ib_dci *const dci =
      pgaspi_ib_dc_dci (&ib_dev_ctx->dciP, rank);

    if (NULL == dci)
    {
      return ENOMEM;
    }

    return pgaspi_ib_dc_post (ib_dev_ctx, dci, rank,
                              ib_dev_ctx->remote_info[rank].qpnP, wr);
  }
#endif

  struct ibv_send_wr *bad_wr;

  return ibv_post_send (ib_dev_ctx->qpP[rank], wr, &bad_wr);
}

#endif // _GPI2_IB_H_
