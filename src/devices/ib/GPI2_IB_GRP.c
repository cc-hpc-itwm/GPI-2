/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2019

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
#include <sys/mman.h>
#include <sys/timeb.h>
#include <unistd.h>
#include "GPI2.h"
#include "GASPI.h"
#include "GPI2_Coll.h"
#include "GPI2_IB.h"

/* Group utilities */
int
pgaspi_dev_poll_groups (gaspi_context_t * const gctx)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  const int pret = ibv_poll_cq (ib_dev_ctx->scqGroups,
                                gctx->ne_count_grp,
                                ib_dev_ctx->wc_grp_send);

  if (pret < 0)
  {
    gaspi_uint i;
    for (i = 0; i < gctx->ne_count_grp; i++)
    {
      /* TODO: wc_grp_send is a [64] so we're basically assuming
         that ne_count_grp will never exceed that */
      if (ib_dev_ctx->wc_grp_send[i].status != IBV_WC_SUCCESS)
      {
        //TODO: for now here because we need to identify the erroneous rank
        // but has to go out of device
        gctx->state_vec[GASPI_COLL_QP][ib_dev_ctx->wc_grp_send[i].wr_id] =
          GASPI_STATE_CORRUPT;
      }
    }

    gaspi_debug_print_error
      ("Failed request to %lu. Collectives queue might be broken",
       ib_dev_ctx->wc_grp_send[i].wr_id);
    return -1;
  }

  gctx->ne_count_grp -= pret;

  return pret;
}

int
pgaspi_dev_post_group_write (gaspi_context_t * const gctx,
                             void *local_addr, int length, int dst,
                             void *remote_addr, unsigned char group)
{
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  struct ibv_send_wr *bad_wr_send;

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

  if (ibv_post_send
      ((struct ibv_qp *) ib_dev_ctx->qpGroups[dst], &swr, &bad_wr_send))
  {
    return 1;
  }

  gctx->ne_count_grp++;

  return 0;
}
