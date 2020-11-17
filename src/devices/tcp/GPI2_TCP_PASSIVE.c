/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2020

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

gaspi_return_t
pgaspi_dev_passive_send (gaspi_context_t * const gctx,
                         const gaspi_segment_id_t segment_id,
                         const gaspi_offset_t offset_local,
                         const gaspi_rank_t rank,
                         const gaspi_size_t size,
                         const gaspi_timeout_t timeout_ms)
{
  gaspi_cycles_t s0;

  const int byte_id = rank >> 3;
  const int bit_pos = rank - (byte_id * 8);
  const unsigned char bit_cmp = 1 << bit_pos;

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  if (gctx->ne_count_p[byte_id] & bit_cmp)
  {
    goto checkL;
  }

  tcp_dev_wr_t wr =
    {
      .wr_id = rank,
      .cq_handle = tcp_dev_ctx->scqP->num,
      .source = gctx->rank,
      .target = rank,
      .local_addr =
      (uintptr_t) (gctx->rrmd[segment_id][gctx->rank].data.addr +
                   offset_local),
      .remote_addr = 0UL,
      .length = size,
      .swap = 0,
      .compare_add = 0,
      .opcode = POST_SEND
    };

  if (write (tcp_dev_ctx->qpP->handle, &wr, sizeof (tcp_dev_wr_t)) <
      (ssize_t) sizeof (tcp_dev_wr_t))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_p[byte_id] |= bit_cmp;

checkL:
  s0 = gaspi_get_cycles();

  int ne = 0;
  tcp_dev_wc_t wc;

  do
  {
    ne = tcp_dev_return_wc (tcp_dev_ctx->scqP, &wc);

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

  if ((ne < 0) || (wc.status != TCP_WC_SUCCESS))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_p[byte_id] &= (~bit_cmp);

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_passive_receive (gaspi_context_t * const gctx,
                            const gaspi_segment_id_t segment_id_local,
                            const gaspi_offset_t offset_local,
                            gaspi_rank_t * const rem_rank,
                            const gaspi_size_t size,
                            const gaspi_timeout_t timeout_ms)
{
  fd_set rfds;
  struct timeval tout;
  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  tcp_dev_wr_t wr =
    {
      .wr_id = gctx->rank,
      .cq_handle = tcp_dev_ctx->rcqP->num,
      .source = gctx->rank,
      .target = 0,
      .local_addr =
      (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
                   offset_local),
      .remote_addr = 0UL,
      .length = size,
      .swap = 0,
      .compare_add = 0,
      .opcode = POST_RECV
    };

  if (write (tcp_dev_ctx->srqP, &wr, sizeof (tcp_dev_wr_t)) <
      (ssize_t) sizeof (tcp_dev_wr_t))
  {
    return GASPI_ERROR;
  }

  FD_ZERO (&rfds);
  FD_SET (tcp_dev_ctx->channelP->read, &rfds);

  //TODO: repeated code
  const long ts = (timeout_ms / 1000);
  const long tus = (timeout_ms - ts * 1000) * 1000;

  tout.tv_sec = ts;
  tout.tv_usec = tus;

  const int selret = select (FD_SETSIZE, &rfds, NULL, NULL, &tout);

  if (selret < 0)
  {
    return GASPI_ERROR;
  }
  else if (selret == 0)
  {
    return GASPI_TIMEOUT;
  }

  /* ack returned event */
  {
    char buf;

    if (read (tcp_dev_ctx->channelP->read, &buf, 1) < 0)
    {
      return GASPI_ERROR;
    }
  }

  int ne = 0;
  tcp_dev_wc_t wc;

  do
  {
    ne = tcp_dev_return_wc (tcp_dev_ctx->rcqP, &wc);
  }
  while (ne == 0);

  if ((ne < 0) || (wc.status != TCP_WC_SUCCESS))
  {
    gctx->state_vec[GASPI_PASSIVE_QP][wc.wr_id] = GASPI_STATE_CORRUPT;
    return GASPI_ERROR;
  }

  /* set sender rank */
  *rem_rank = (gaspi_rank_t) wc.sender;

  return GASPI_SUCCESS;
}
