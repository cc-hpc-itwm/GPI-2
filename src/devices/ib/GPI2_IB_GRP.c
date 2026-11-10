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
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>
#include "GASPI_types.h"
#include "GPI2_GRP.h"
#include "GPI2_IB.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

/* Group utilities */
int
pgaspi_dev_poll_groups (gaspi_context_t * const gctx)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  const int max = (int) MIN (gctx->ne_count_grp,
                             sizeof (ib_dev_ctx->wc_grp_send)
                             / sizeof (ib_dev_ctx->wc_grp_send[0]));
  const int pret = ibv_poll_cq (ib_dev_ctx->scqGroups, max,
                                ib_dev_ctx->wc_grp_send);

  if (pret < 0)
  {
    for (int i = 0; i < max; i++)
    {
      if (ib_dev_ctx->wc_grp_send[i].status != IBV_WC_SUCCESS)
      {
        //TODO: for now here because we need to identify the erroneous rank
        // but has to go out of device
        gctx->state_vec[GASPI_COLL_QP][ib_dev_ctx->wc_grp_send[i].wr_id] =
          GASPI_STATE_CORRUPT;
      }
    }

    GASPI_DEBUG_PRINT_ERROR ("Collectives queue might be broken");
    return -1;
  }

  __atomic_sub_fetch (&gctx->ne_count_grp, pret, __ATOMIC_RELAXED);

  /* DC: a failed request moves its groups DCI to the error state and the
     requests behind it, also to other ranks, are flushed. Mark the rank of
     the failed one and recover the DCI; the collective then times out, as
     with RC, where the failed requests are not checked here. */
#ifdef GPI2_IB_DC
  if (ib_dev_ctx->dc)
  {
    int failed = 0;

    for (int i = 0; i < pret; i++)
    {
      if (ib_dev_ctx->wc_grp_send[i].status != IBV_WC_SUCCESS)
      {
        if (ib_dev_ctx->wc_grp_send[i].status != IBV_WC_WR_FLUSH_ERR)
        {
          gctx->state_vec[GASPI_COLL_QP][ib_dev_ctx->wc_grp_send[i].wr_id] =
            GASPI_STATE_CORRUPT;
        }
        failed = 1;
      }
    }

    if (failed)
    {
      /* Only what is outstanding now: other threads may post meanwhile */
      const gaspi_uint n =
        __atomic_load_n (&gctx->ne_count_grp, __ATOMIC_RELAXED);
      gaspi_uint left = n;
      const int rc = pgaspi_ib_dc_recover (gctx, &ib_dev_ctx->dciGroups,
                                           ib_dev_ctx->scqGroups, &left);

      __atomic_sub_fetch (&gctx->ne_count_grp, n - left, __ATOMIC_RELAXED);
      if (rc != 0)
      {
        return -1;
      }
    }
  }
#endif

  return pret;
}

int
pgaspi_dev_post_group_write (gaspi_context_t * const gctx,
                             void *local_addr, int length, int dst,
                             void *remote_addr, unsigned char group)
{
  struct ibv_sge slist;
  struct ibv_send_wr swr;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  slist.addr = (uintptr_t) local_addr;
  slist.length = length;
  slist.lkey =
    ((struct ibv_mr *) gctx->groups[group].rrcd[gctx->rank].mr[0])->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags =
    (length == 1) ? (IBV_SEND_SIGNALED | IBV_SEND_INLINE) : IBV_SEND_SIGNALED;
  swr.next = NULL;

  swr.wr.rdma.remote_addr = (uint64_t) remote_addr;
  swr.wr.rdma.rkey = gctx->groups[group].rrcd[dst].rkey[0];
  swr.wr_id = dst;

  if (pgaspi_ib_post_groups (ib_dev_ctx, dst, &swr))
  {
    return 1;
  }

  __atomic_add_fetch (&gctx->ne_count_grp, 1, __ATOMIC_RELAXED);

  return 0;
}

/* Post receive requests without buffers on the groups QP to rank (DC: on
   the SRQ of the groups DCT, shared by all peers): a write with immediate
   data consumes one but places its data directly. */
int
pgaspi_ib_post_group_recvs (gaspi_context_t const *const gctx,
                            const int rank, const int count)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_recv_wr rwr;
  struct ibv_recv_wr *bad_rwr;

  rwr.wr_id = rank;
  rwr.sg_list = NULL;
  rwr.num_sge = 0;
  rwr.next = NULL;

#ifdef GPI2_IB_DC
  if (ib_dev_ctx->dc)
  {
    for (int i = 0; i < count; i++)
    {
      if (ibv_post_srq_recv (ib_dev_ctx->srqGroups, &rwr, &bad_rwr))
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed to post receive (libibverbs)");
        return -1;
      }
    }

    return 0;
  }
#endif

  for (int i = 0; i < count; i++)
  {
    if (ibv_post_recv (ib_dev_ctx->qpGroups[rank], &rwr, &bad_rwr))
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to post receive (libibverbs)");
      return -1;
    }
  }

  return 0;
}

int
pgaspi_dev_post_group_write_sync (gaspi_context_t * const gctx,
                                  void *local_addr, int length, int dst,
                                  void *remote_addr,
                                  void *flag_local_addr,
                                  void *flag_remote_addr,
                                  uint32_t imm, unsigned char group)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (!ib_dev_ctx->relaxed_ordering)
  {
    if (pgaspi_dev_post_group_write (gctx, local_addr, length, dst,
                                     remote_addr, group) != 0)
    {
      return 1;
    }

    return pgaspi_dev_post_group_write (gctx, flag_local_addr, 1, dst,
                                        flag_remote_addr, group);
  }

  /* With relaxed ordering, the flag may become visible before the data if
     both are written with RDMA writes. Instead, the data is written with
     the flag as immediate data: the receiver sets the flag once it polls
     the completion, which guarantees the visibility of the data (IB spec
     o9-20). */
  struct ibv_sge slist;
  struct ibv_send_wr swr;

  slist.addr = (uintptr_t) local_addr;
  slist.length = length;
  slist.lkey =
    ((struct ibv_mr *) gctx->groups[group].rrcd[gctx->rank].mr[0])->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.imm_data = htonl (imm);
  swr.next = NULL;

  swr.wr.rdma.remote_addr = (uint64_t) remote_addr;
  swr.wr.rdma.rkey = gctx->groups[group].rrcd[dst].rkey[0];
  swr.wr_id = dst;

  if (pgaspi_ib_post_groups (ib_dev_ctx, dst, &swr))
  {
    return 1;
  }

  __atomic_add_fetch (&gctx->ne_count_grp, 1, __ATOMIC_RELAXED);

  return 0;
}

/* Polled independently of the local relaxed ordering setting, as a peer
   may have it set (see pgaspi_dev_create_endpoint). */
int
pgaspi_dev_poll_group_sync (gaspi_context_t * const gctx)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_wc wc[GPI2_IB_GRP_RECV_DEPTH];

  const int ne =
    ibv_poll_cq (ib_dev_ctx->rcqGroups, GPI2_IB_GRP_RECV_DEPTH, wc);

  if (ne < 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to poll groups receive CQ");
    return -1;
  }

  for (int i = 0; i < ne; i++)
  {
    /* RC: wr_id is the rank of the QP the receive was posted on, and it is
       valid in a failed completion. DC: the receives on the shared SRQ are
       posted with wr_id 0 and any peer consumes them, so the sender of a
       failed one is unknown and no rank is marked. */
    const int src = (int) wc[i].wr_id;

    /* receive requests of a destroyed QP (disconnected rank) */
    if (wc[i].status == IBV_WC_WR_FLUSH_ERR)
    {
      continue;
    }

    if (wc[i].status != IBV_WC_SUCCESS
        || wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM)
    {
      if (ib_dev_ctx->dc)
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed receive (status %d)", wc[i].status);
      }
      else
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed receive from %d (status %d)",
                                 src, wc[i].status);
        gctx->state_vec[GASPI_COLL_QP][src] = GASPI_STATE_CORRUPT;
      }
      return -1;
    }

    pgaspi_group_sync_flag_set (gctx, ntohl (wc[i].imm_data));

    if (pgaspi_ib_post_group_recvs (gctx, src, 1) != 0)
    {
      if (!ib_dev_ctx->dc)
      {
        gctx->state_vec[GASPI_COLL_QP][src] = GASPI_STATE_CORRUPT;
      }
      return -1;
    }
  }

  return ne;
}
