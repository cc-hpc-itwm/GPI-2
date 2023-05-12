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
#include "GASPI.h"
#include "GPI2_TCP.h"
#include "GPI2_Types.h"

int
pgaspi_dev_post_group_write (gaspi_context_t * const gctx,
                             void *local_addr, int length, int dst,
                             void *remote_addr,
                             unsigned char GASPI_UNUSED (g))
{
  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  tcp_dev_wr_t wr =
    {
      .cq_handle = tcp_dev_ctx->scqGroups->num,
      .source = gctx->rank,
      .local_addr = (uintptr_t) local_addr,
      .length = length,
      .swap = 0,
      .compare_add = 0,
      .opcode = POST_RDMA_WRITE,
      .target = dst,
      .remote_addr = (uintptr_t) remote_addr,
      .wr_id = dst
    };

  if (write (tcp_dev_ctx->qpGroups->handle, &wr, sizeof (tcp_dev_wr_t)) <
      (ssize_t) sizeof (tcp_dev_wr_t))
  {
    return 1;
  }

  gctx->ne_count_grp++;

  return 0;
}

/* TODO: number of elems to poll as arg */
int
pgaspi_dev_poll_groups (gaspi_context_t * const gctx)
{
  const int nr = gctx->ne_count_grp;
  tcp_dev_wc_t wc;

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  int ne = 0;
  for (int i = 0; i < nr; i++)
  {
    do
    {
      ne = tcp_dev_return_wc (tcp_dev_ctx->scqGroups, &wc);
    }
    while (ne == 0);

    if ((ne < 0) || (wc.status != TCP_WC_SUCCESS))
    {
      /* it can be that you encounter an erroneous completion but
         from a peer that is gone. For now, we just ignore it by
         checking the validity of the state */
      if (!tcp_dev_is_valid_state (wc.wr_id))
      {
        continue;
      }

      /* TODO: for now here because of id of erroneous rank, but has to go out of device */
      gctx->state_vec[GASPI_COLL_QP][wc.wr_id] = GASPI_STATE_CORRUPT;

      GASPI_DEBUG_PRINT_ERROR
        ("Failed request to %lu. Collectives queue might be broken", wc.wr_id);
      return -1;
    }
  }

  gctx->ne_count_grp -= nr;

  return nr;
}
