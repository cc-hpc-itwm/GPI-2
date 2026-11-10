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
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include "GASPI_types.h"
#include "GPI2_IB.h"
#include "GPI2_Mem.h"
#include "GPI2_Sys.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

gaspi_return_t
pgaspi_dev_passive_send (gaspi_context_t * const gctx,
                         const gaspi_segment_id_t segment_id_local,
                         const gaspi_offset_t offset_local,
                         const gaspi_rank_t rank,
                         const gaspi_size_t size,
                         const gaspi_timeout_t timeout_ms)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_sge slist;
  struct ibv_send_wr swr;
  struct ibv_wc wc_send;
  gaspi_cycles_t s0;

  const int byte_id = rank >> 3;
  const int bit_pos = rank - (byte_id * 8);
  const unsigned char bit_cmp = 1 << bit_pos;

  if (gctx->ne_count_p[byte_id] & bit_cmp)
  {
    goto checkL;
  }

  slist.addr =
    (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
                 offset_local);
  slist.length = size;
  slist.lkey =
    ((struct ibv_mr *) gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;

  /* DC: all senders reach the same DCT, the receiver learns the sender from
     the immediate data */
#ifdef GPI2_IB_DC
  if (ib_dev_ctx->dc)
  {
    swr.opcode = IBV_WR_SEND_WITH_IMM;
    swr.imm_data = htonl (gctx->rank);
  }
  else
#endif
  swr.opcode = IBV_WR_SEND;
  swr.wr_id = rank;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = NULL;

  if (pgaspi_ib_post_passive (ib_dev_ctx, rank, &swr))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_p[byte_id] |= bit_cmp;

checkL:

  s0 = gaspi_get_cycles();

  int ne = 0;

  do
  {
    ne = ibv_poll_cq (ib_dev_ctx->scqP, 1, &wc_send);

    if (ne == 0)
    {
      const gaspi_cycles_t s1 = gaspi_get_cycles();
      const gaspi_cycles_t tdelta = s1 - s0;

      const float ms = (float) tdelta * gctx->cycles_to_msecs;

      if (ms > timeout_ms)
      {
        return GASPI_TIMEOUT;
      }
    }
  }
  while (ne == 0);

  if ((ne < 0) || (wc_send.status != IBV_WC_SUCCESS))
  {
    /* DC: recover the passive DCI (a send to another rank still pending on
       it, after a timeout, is lost) */
#ifdef GPI2_IB_DC
    if (ib_dev_ctx->dc)
    {
      gaspi_uint behind = 0;

      pgaspi_ib_dc_recover (gctx, &ib_dev_ctx->dciP, ib_dev_ctx->scqP,
                            &behind);
    }
#endif
    return GASPI_ERROR;
  }

  gctx->ne_count_p[byte_id] &= (~bit_cmp);

  return GASPI_SUCCESS;
}

static int
ib_passive_post_slot (gaspi_ib_ctx * const ib_dev_ctx, const uint32_t slot)
{
  struct ibv_sge rlist;
  struct ibv_recv_wr rwr;
  struct ibv_recv_wr *bad_wr;

  rlist.addr = (uintptr_t) ((char *) ib_dev_ctx->passive_buf
                            + (size_t) slot * ib_dev_ctx->passive_slot_size);
  rlist.length = ib_dev_ctx->passive_slot_size;
  rlist.lkey = ib_dev_ctx->passive_mr->lkey;

  rwr.wr_id = slot;
  rwr.sg_list = &rlist;
  rwr.num_sge = 1;
  rwr.next = NULL;

  return ibv_post_srq_recv (ib_dev_ctx->srqP, &rwr, &bad_wr);
}

int
pgaspi_ib_passive_init (gaspi_context_t const *const gctx)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  ib_dev_ctx->passive_slots =
    MIN (GPI2_IB_PASSIVE_SLOTS, gctx->config->queue_size_max);
  ib_dev_ctx->passive_slot_size =
    (gctx->config->passive_transfer_size_max + 63) & ~63u;

  const size_t bytes =
    (size_t) ib_dev_ctx->passive_slots * ib_dev_ctx->passive_slot_size;

  if (pgaspi_alloc_page_aligned (&ib_dev_ctx->passive_buf, bytes) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Memory allocation (passive slots) failed");
    return -1;
  }

  ib_dev_ctx->passive_mr = ibv_reg_mr (ib_dev_ctx->pd,
                                       ib_dev_ctx->passive_buf, bytes,
                                       IBV_ACCESS_LOCAL_WRITE);
  if (NULL == ib_dev_ctx->passive_mr)
  {
    GASPI_DEBUG_PRINT_ERROR ("Memory registration failed (libibverbs)");
    return -1;
  }

  for (uint32_t slot = 0; slot < ib_dev_ctx->passive_slots; slot++)
  {
    if (ib_passive_post_slot (ib_dev_ctx, slot))
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to post passive receive (libibverbs)");
      return -1;
    }
  }

  return 0;
}

int
pgaspi_ib_passive_cleanup (gaspi_ib_ctx * const ib_dev_ctx)
{
  if (ib_dev_ctx->passive_mr != NULL && ibv_dereg_mr (ib_dev_ctx->passive_mr))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to de-register memory (libibverbs)");
    return -1;
  }
  ib_dev_ctx->passive_mr = NULL;

  free (ib_dev_ctx->passive_buf);
  ib_dev_ctx->passive_buf = NULL;

  return 0;
}

static gaspi_rank_t
ib_passive_rc_sender (gaspi_context_t const *const gctx,
                      gaspi_ib_ctx const *const ib_dev_ctx,
                      const uint32_t qp_num)
{
  for (gaspi_rank_t i = 0; i < gctx->tnc; i++)
  {
    /* we need to make sure the QP was already created and valid */
    if (ib_dev_ctx->qpP != NULL
        && ib_dev_ctx->qpP[i] != NULL
        && ib_dev_ctx->qpP[i]->qp_num == qp_num)
    {
      return i;
    }
  }

  return 0xffff;
}

gaspi_return_t
pgaspi_dev_passive_receive (gaspi_context_t * const gctx,
                            const gaspi_segment_id_t segment_id_local,
                            const gaspi_offset_t offset_local,
                            gaspi_rank_t * const rem_rank,
                            const gaspi_size_t size,
                            const gaspi_timeout_t timeout_ms)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  struct ibv_wc wc_recv;
  const gaspi_cycles_t s0 = gaspi_get_cycles();

  int ne = ibv_poll_cq (ib_dev_ctx->rcqP, 1, &wc_recv);

  while (ne == 0)
  {
    if (ibv_req_notify_cq (ib_dev_ctx->rcqP, 0))
    {
      return GASPI_ERROR;
    }

    ne = ibv_poll_cq (ib_dev_ctx->rcqP, 1, &wc_recv);
    if (ne != 0)
    {
      break;
    }

    const float ms =
      (float) (gaspi_get_cycles() - s0) * gctx->cycles_to_msecs;

    if (ms >= (float) timeout_ms)
    {
      return GASPI_TIMEOUT;
    }

    const gaspi_timeout_t left = timeout_ms - (gaspi_timeout_t) ms;
    struct timeval tout;

    tout.tv_sec = left / 1000;
    tout.tv_usec = (left % 1000) * 1000;

    fd_set rfds;

    FD_ZERO (&rfds);
    FD_SET (ib_dev_ctx->channelP->fd, &rfds);

    const int selret =
      select (ib_dev_ctx->channelP->fd + 1, &rfds, NULL, NULL, &tout);

    if (selret < 0)
    {
      return GASPI_ERROR;
    }
    else if (selret == 0)
    {
      return GASPI_TIMEOUT;
    }

    struct ibv_cq *ev_cq;
    void *ev_ctx;

    if (ibv_get_cq_event (ib_dev_ctx->channelP, &ev_cq, &ev_ctx))
    {
      return GASPI_ERROR;
    }

    ibv_ack_cq_events (ev_cq, 1);

    if (ev_cq != ib_dev_ctx->rcqP)
    {
      return GASPI_ERROR;
    }

    ne = ibv_poll_cq (ib_dev_ctx->rcqP, 1, &wc_recv);
  }

  if (ne < 0)
  {
    return GASPI_ERROR;
  }

  const uint32_t slot = (uint32_t) wc_recv.wr_id;

  if (wc_recv.status != IBV_WC_SUCCESS)
  {
    const gaspi_rank_t sender =
      ib_passive_rc_sender (gctx, ib_dev_ctx, wc_recv.qp_num);

    //TODO: for now here but has to go up
    if (sender != 0xffff)
    {
      gctx->state_vec[GASPI_PASSIVE_QP][sender] = GASPI_STATE_CORRUPT;
    }

    ib_passive_post_slot (ib_dev_ctx, slot);
    return GASPI_ERROR;
  }

  gaspi_rank_t sender;

#ifdef GPI2_IB_DC
  if (ib_dev_ctx->dc)
  {
    sender = (gaspi_rank_t) ntohl (wc_recv.imm_data);
  }
  else
#endif
  {
    do
    {
      sender = ib_passive_rc_sender (gctx, ib_dev_ctx, wc_recv.qp_num);
    }
    while (sender == 0xffff);
  }

  gaspi_return_t eret = GASPI_SUCCESS;

  if (sender >= gctx->tnc)
  {
    eret = GASPI_ERROR;
  }
  else if (wc_recv.byte_len > size)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Passive message of %u bytes for a receive of %lu",
       wc_recv.byte_len, (unsigned long) size);
    eret = GASPI_ERROR;
  }
  else
  {
    memcpy ((void *) (gctx->rrmd[segment_id_local][gctx->rank].data.addr
                      + offset_local),
            (char *) ib_dev_ctx->passive_buf
            + (size_t) slot * ib_dev_ctx->passive_slot_size,
            wc_recv.byte_len);
    *rem_rank = sender;
  }

  if (ib_passive_post_slot (ib_dev_ctx, slot))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to post passive receive (libibverbs)");
    return GASPI_ERROR;
  }

  return eret;
}
