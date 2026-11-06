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
/* DC initiator: one per channel, it reaches any DCT */
struct pgaspi_ib_dci
{
  struct ibv_qp *qp;
  struct ibv_qp_ex *qpx;
  struct mlx5dv_qp_ex *mqpx;
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

  /* Dynamically Connected transport (GASPI_IB_DCT): one initiator (DCI) and
     one target (DCT) per channel instead of an RC QP per channel and peer */
  int dc;
#ifdef GPI2_IB_DC
  struct pgaspi_ib_dci dciGroups, dciP, dciC[GASPI_MAX_QP];
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
#endif

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
    return pgaspi_ib_dc_post (ib_dev_ctx, &ib_dev_ctx->dciC[queue], rank,
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
    return pgaspi_ib_dc_post (ib_dev_ctx, &ib_dev_ctx->dciGroups, rank,
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
    return pgaspi_ib_dc_post (ib_dev_ctx, &ib_dev_ctx->dciP, rank,
                              ib_dev_ctx->remote_info[rank].qpnP, wr);
  }
#endif

  struct ibv_send_wr *bad_wr;

  return ibv_post_send (ib_dev_ctx->qpP[rank], wr, &bad_wr);
}

#endif // _GPI2_IB_H_
