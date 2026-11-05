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

#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>
#include "GASPI_types.h"
#include "GPI2_IB.h"
#include "GPI2_Types.h"

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

  struct ibv_send_wr swr;

  swr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  swr.send_flags = IBV_SEND_SIGNALED;

  swr.wr.atomic.remote_addr = gctx->rrmd[segment_id][rank].data.addr + offset;
  swr.wr.atomic.rkey = gctx->rrmd[segment_id][rank].rkey[0];
  swr.wr.atomic.compare_add = val_add;

  swr.wr_id = rank;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (pgaspi_ib_post_groups (ib_dev_ctx, rank, &swr))
  {
    return GASPI_ERROR;
  }

  int nr = __atomic_exchange_n (&gctx->ne_count_grp, 0, __ATOMIC_RELAXED) + 1;

  int ne = 0;

  for (int i = 0; i < nr; i++)
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

  struct ibv_send_wr swr;

  swr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  swr.send_flags = IBV_SEND_SIGNALED;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  swr.wr.atomic.remote_addr = gctx->rrmd[segment_id][rank].data.addr + offset;
  swr.wr.atomic.rkey = gctx->rrmd[segment_id][rank].rkey[0];
  swr.wr.atomic.compare_add = comparator;
  swr.wr.atomic.swap = val_new;

  swr.wr_id = rank;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.next = NULL;

  if (pgaspi_ib_post_groups (ib_dev_ctx, rank, &swr))
  {
    return GASPI_ERROR;
  }

  int nr = __atomic_exchange_n (&gctx->ne_count_grp, 0, __ATOMIC_RELAXED) + 1;

  int ne = 0;

  for (int i = 0; i < nr; i++)
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

  return GASPI_SUCCESS;
}
