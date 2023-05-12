/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2023

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
#include "GPI2.h"
#include "GASPI.h"
#include "GPI2_IB.h"

gaspi_return_t
pgaspi_dev_atomic_fetch_add (gaspi_context_t * const gctx,
                             const gaspi_segment_id_t segment_id,
                             const gaspi_offset_t offset,
                             const gaspi_rank_t rank,
                             const gaspi_atomic_value_t val_add)
{
  struct ibv_sge slist;

  slist.addr = (uintptr_t) (gctx->nsrc.data.buf);
  slist.length = sizeof (gaspi_atomic_value_t);
  slist.lkey = ((struct ibv_mr *) gctx->nsrc.mr[0])->lkey;

#ifdef GPI2_EXP_VERBS
  struct ibv_exp_send_wr *bad_wr;
  struct ibv_exp_send_wr swr;

  swr.exp_opcode = IBV_EXP_WR_ATOMIC_FETCH_AND_ADD;
  swr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
#else
  struct ibv_send_wr *bad_wr;
  struct ibv_send_wr swr;

  swr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  swr.send_flags = IBV_SEND_SIGNALED;
#endif

  swr.wr.atomic.remote_addr = gctx->rrmd[segment_id][rank].data.addr + offset;
  swr.wr.atomic.rkey = gctx->rrmd[segment_id][rank].rkey[0];
  swr.wr.atomic.compare_add = val_add;

  swr.wr_id = rank;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

#ifdef GPI2_EXP_VERBS
  if (ibv_exp_post_send (ib_dev_ctx->qpGroups[rank], &swr, &bad_wr))
  {
    return GASPI_ERROR;
  }
#else
  if (ibv_post_send (ib_dev_ctx->qpGroups[rank], &swr, &bad_wr))
  {
    return GASPI_ERROR;
  }
#endif

  //TODO
  gctx->ne_count_grp++;

  int ne = 0;

  for (gaspi_uint i = 0; i < gctx->ne_count_grp; i++)
  {
    do
    {
      ne = ibv_poll_cq (ib_dev_ctx->scqGroups, 1, ib_dev_ctx->wc_grp_send);
    }
    while (ne == 0);

    if ((ne < 0) || (ib_dev_ctx->wc_grp_send[i].status != IBV_WC_SUCCESS))
    {
      return GASPI_ERROR;
    }
  }

  //TODO: remove from here
  gctx->ne_count_grp = 0;

  return GASPI_SUCCESS;
}


gaspi_return_t
pgaspi_dev_atomic_compare_swap (gaspi_context_t * const gctx,
                                const gaspi_segment_id_t segment_id,
                                const gaspi_offset_t offset,
                                const gaspi_rank_t rank,
                                const gaspi_atomic_value_t comparator,
                                const gaspi_atomic_value_t val_new)
{
  struct ibv_sge slist;

  slist.addr = (uintptr_t) (gctx->nsrc.data.buf);
  slist.length = sizeof (gaspi_atomic_value_t);
  slist.lkey = ((struct ibv_mr *) gctx->nsrc.mr[0])->lkey;

#ifdef GPI2_EXP_VERBS
  struct ibv_exp_send_wr *bad_wr;
  struct ibv_exp_send_wr swr;

  swr.exp_opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  swr.exp_send_flags = IBV_SEND_SIGNALED;

#else
  struct ibv_send_wr *bad_wr;
  struct ibv_send_wr swr;

  swr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  swr.send_flags = IBV_SEND_SIGNALED;
#endif

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  swr.wr.atomic.remote_addr = gctx->rrmd[segment_id][rank].data.addr + offset;
  swr.wr.atomic.rkey = gctx->rrmd[segment_id][rank].rkey[0];
  swr.wr.atomic.compare_add = comparator;
  swr.wr.atomic.swap = val_new;

  swr.wr_id = rank;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.next = NULL;

#ifdef GPI2_EXP_VERBS
  if (ibv_exp_post_send (ib_dev_ctx->qpGroups[rank], &swr, &bad_wr))
  {
    return GASPI_ERROR;
  }
#else
  if (ibv_post_send (ib_dev_ctx->qpGroups[rank], &swr, &bad_wr))
  {
    return GASPI_ERROR;
  }
#endif

  gctx->ne_count_grp++;

  int ne = 0;

  for (gaspi_uint i = 0; i < gctx->ne_count_grp; i++)
  {
    do
    {
      ne = ibv_poll_cq (ib_dev_ctx->scqGroups, 1, ib_dev_ctx->wc_grp_send);
    }
    while (ne == 0);

    if ((ne < 0) || (ib_dev_ctx->wc_grp_send[i].status != IBV_WC_SUCCESS))
    {
      return GASPI_ERROR;
    }
  }

  //TODO:
  gctx->ne_count_grp = 0;

  return GASPI_SUCCESS;
}
