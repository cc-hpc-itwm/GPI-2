/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2017

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
#include "GPI2_IB.h"

gaspi_return_t
pgaspi_dev_passive_send (gaspi_context_t * const gctx,
			 const gaspi_segment_id_t segment_id_local,
			 const gaspi_offset_t offset_local,
			 const gaspi_rank_t rank,
			 const gaspi_size_t size,
			 const gaspi_timeout_t timeout_ms)
{
  gaspi_ib_ctx * const ib_dev_ctx = (gaspi_ib_ctx*) gctx->device->ctx;

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  struct ibv_wc wc_send;
  gaspi_cycles_t s0;

  const int byte_id = rank >> 3;
  const int bit_pos = rank - (byte_id * 8);
  const unsigned char bit_cmp = 1 << bit_pos;

  if( gctx->ne_count_p[byte_id] & bit_cmp )
    {
      goto checkL;
    }

  slist.addr = (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
			    offset_local);
  slist.length = size;
  slist.lkey = ((struct ibv_mr *) gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_SEND;
  swr.wr_id = rank;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = NULL;


  if( ibv_post_send (ib_dev_ctx->qpP[rank], &swr, &bad_wr) )
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_p[byte_id] |= bit_cmp;

 checkL:

  s0 = gaspi_get_cycles ();

  int ne = 0;
  do
    {
      ne = ibv_poll_cq (ib_dev_ctx->scqP, 1, &wc_send);

      if (ne == 0)
	{
	  const gaspi_cycles_t s1 = gaspi_get_cycles ();
	  const gaspi_cycles_t tdelta = s1 - s0;

	  const float ms = (float) tdelta * gctx->cycles_to_msecs;
	  if (ms > timeout_ms)
	    {
	      return GASPI_TIMEOUT;
	    }
	}
    }
  while (ne == 0);

  if( (ne < 0) || (wc_send.status != IBV_WC_SUCCESS) )
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
			    gaspi_rank_t * const rem_rank, const gaspi_size_t size,
			    const gaspi_timeout_t timeout_ms)
{

  struct ibv_recv_wr *bad_wr;
  struct ibv_wc wc_recv;
  struct ibv_sge rlist;
  struct ibv_recv_wr rwr;
  struct ibv_cq *ev_cq;
  void *ev_ctx;
  int i;
  fd_set rfds;
  struct timeval tout;

  rlist.addr = (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
			    offset_local);
  rlist.length = size;
  rlist.lkey = ((struct ibv_mr *) gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;
  rwr.wr_id = gctx->rank;
  rwr.sg_list = &rlist;
  rwr.num_sge = 1;
  rwr.next = NULL;

  gaspi_ib_ctx * const ib_dev_ctx = (gaspi_ib_ctx*) gctx->device->ctx;

  if (ibv_post_srq_recv (ib_dev_ctx->srqP, &rwr, &bad_wr))
    {
      return GASPI_ERROR;
    }

  FD_ZERO (&rfds);
  FD_SET (ib_dev_ctx->channelP->fd, &rfds);

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

  if (ibv_get_cq_event (ib_dev_ctx->channelP, &ev_cq, &ev_ctx))
    {
      return GASPI_ERROR;
    }

  ibv_ack_cq_events (ev_cq, 1);

  if (ev_cq != ib_dev_ctx->rcqP)
    {
      return GASPI_ERROR;
    }

  if (ibv_req_notify_cq (ib_dev_ctx->rcqP, 0))
    {
      return GASPI_ERROR;
    }

  int ne = 0;
  do
    {
      ne = ibv_poll_cq (ib_dev_ctx->rcqP, 1, &wc_recv);
    }
  while (ne == 0);

  if ((ne < 0) || (wc_recv.status != IBV_WC_SUCCESS))
    {
      //TODO: for now here but has to go up
      gctx->qp_state_vec[GASPI_PASSIVE_QP][wc_recv.wr_id] = GASPI_STATE_CORRUPT;
      return GASPI_ERROR;
    }

  *rem_rank = 0xffff;
  do
    {
      for (i = 0; i < gctx->tnc; i++)
	{
	  /* we need to make sure the QP was already created and valid */
	  if(ib_dev_ctx->qpP != NULL)
	    if(ib_dev_ctx->qpP[i] != NULL)
	      if (ib_dev_ctx->qpP[i]->qp_num == wc_recv.qp_num)
		{
		  *rem_rank = i;
		  break;
		}
	}
    }
  while(*rem_rank == 0xffff);

  return GASPI_SUCCESS;
}
