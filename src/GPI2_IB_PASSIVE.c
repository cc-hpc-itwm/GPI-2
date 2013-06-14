/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013

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

gaspi_return_t
pgaspi_passive_send (const gaspi_segment_id_t segment_id_local,
		    const gaspi_offset_t offset_local,
		    const gaspi_rank_t rank, const gaspi_size_t size,
		    const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG  
  if (glb_gaspi_ctx_ib.rrmd[segment_id_local] == NULL)
    {
      gaspi_printf("Debug: Invalid local segment (gaspi_passive_send)\n");    
      return GASPI_ERROR;
    }
  
  if( rank >= glb_gaspi_ctx.tnc)
    {
      gaspi_printf("Debug: Invalid rank (gaspi_passive_send)\n");    
      return GASPI_ERROR;
    }
  
  if( offset_local > glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].size)
    {
      gaspi_printf("Debug: Invalid offsets (gaspi_passive_send)\n");    
      return GASPI_ERROR;
    }
    
  if( size < 1 || size > GASPI_MAX_TSIZE_P )
    {
      gaspi_printf("Debug: Invalid size (gaspi_passive_send)\n");    
      return GASPI_ERROR;
    }
#endif

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  struct ibv_wc wc_send;
  gaspi_cycles_t s0;

  lock_gaspi_tout (&glb_gaspi_ctx.lockPS, timeout_ms);

  const int byte_id = rank >> 3;
  const int bit_pos = rank - (byte_id * 8);
  const unsigned char bit_cmp = 1 << bit_pos;
  if (glb_gaspi_ctx_ib.ne_count_p[byte_id] & bit_cmp)
    goto checkL;

  slist.addr =
    (uintptr_t) (glb_gaspi_ctx_ib.
		 rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		 NOTIFY_OFFSET + offset_local);
  slist.length = size;
  slist.lkey =
    glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].mr->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_SEND;
  swr.wr_id = rank;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpP[rank], &swr, &bad_wr))
    {
      glb_gaspi_ctx.qp_state_vec[GASPI_PASSIVE_QP][rank] = 1;
      unlock_gaspi (&glb_gaspi_ctx.lockPS);
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_p[byte_id] |= bit_cmp;

checkL:

  s0 = gaspi_get_cycles ();

  int ne = 0;
  do
    {
      ne = ibv_poll_cq (glb_gaspi_ctx_ib.scqP, 1, &wc_send);

      if (ne == 0)
	{
	  const gaspi_cycles_t s1 = gaspi_get_cycles ();
	  const gaspi_cycles_t tdelta = s1 - s0;

	  const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
	  if (ms > timeout_ms)
	    {
	      unlock_gaspi (&glb_gaspi_ctx.lockPS);
	      return GASPI_TIMEOUT;
	    }
	}

    }
  while (ne == 0);

  if ((ne < 0) || (wc_send.status != IBV_WC_SUCCESS))
    {
      glb_gaspi_ctx.qp_state_vec[GASPI_PASSIVE_QP][wc_send.wr_id] = 1;
      unlock_gaspi (&glb_gaspi_ctx.lockPS);
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_p[byte_id] &= (~bit_cmp);

  unlock_gaspi (&glb_gaspi_ctx.lockPS);
  return GASPI_SUCCESS;
}


gaspi_return_t
pgaspi_passive_receive (const gaspi_segment_id_t segment_id_local,
		       const gaspi_offset_t offset_local,
		       gaspi_rank_t * const rem_rank, const gaspi_size_t size,
		       const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG  
  if (glb_gaspi_ctx_ib.rrmd[segment_id_local] == NULL)
    {
      gaspi_printf("Debug: Invalid local segment (gaspi_passive_receive)\n");    
      return GASPI_ERROR;
    }
  
  if( rem_rank == NULL)
    {
      gaspi_printf("Debug: Invalid pointer parameter: rem_rank (gaspi_passive_receive)\n");    
      return GASPI_ERROR;
    }
  
  if( offset_local > glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].size)
    {
      gaspi_printf("Debug: Invalid offsets (gaspi_passive_receive)\n");    
      return GASPI_ERROR;
    }
    
  if( size < 1 || size > GASPI_MAX_TSIZE_P )
    {
      gaspi_printf("Debug: Invalid size (gaspi_passive_receive)\n");    
      return GASPI_ERROR;
    }
#endif

  struct ibv_recv_wr *bad_wr;
  struct ibv_wc wc_recv;
  struct ibv_sge rlist;
  struct ibv_recv_wr rwr;
  struct ibv_cq *ev_cq;
  void *ev_ctx;
  int i;
  fd_set rfds;
  struct timeval tout;


  lock_gaspi_tout (&glb_gaspi_ctx.lockPR, timeout_ms);

  rlist.addr =
    (uintptr_t) (glb_gaspi_ctx_ib.
		 rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		 NOTIFY_OFFSET + offset_local);
  rlist.length = size;
  rlist.lkey =
    glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].mr->lkey;
  rwr.wr_id = glb_gaspi_ctx.rank;
  rwr.sg_list = &rlist;
  rwr.num_sge = 1;
  rwr.next = NULL;

  if (ibv_post_srq_recv (glb_gaspi_ctx_ib.srqP, &rwr, &bad_wr))
    {
      unlock_gaspi (&glb_gaspi_ctx.lockPR);
      return GASPI_ERROR;
    }

  FD_ZERO (&rfds);
  FD_SET (glb_gaspi_ctx_ib.channelP->fd, &rfds);

  const long ts = (timeout_ms / 1000);
  const long tus = (timeout_ms - ts * 1000) * 1000;

  tout.tv_sec = ts;
  tout.tv_usec = tus;

  const int selret = select (FD_SETSIZE, &rfds, NULL, NULL, &tout);
  if (selret < 0)
    {
      unlock_gaspi (&glb_gaspi_ctx.lockPR);
      return GASPI_ERROR;
    }
  else if (selret == 0)
    {
      unlock_gaspi (&glb_gaspi_ctx.lockPR);
      return GASPI_TIMEOUT;
    }

  if (ibv_get_cq_event (glb_gaspi_ctx_ib.channelP, &ev_cq, &ev_ctx))
    {
      unlock_gaspi (&glb_gaspi_ctx.lockPR);
      return GASPI_ERROR;
    }

  ibv_ack_cq_events (ev_cq, 1);

  if (ev_cq != glb_gaspi_ctx_ib.rcqP)
    {
      unlock_gaspi (&glb_gaspi_ctx.lockPR);
      return GASPI_ERROR;
    }

  if (ibv_req_notify_cq (glb_gaspi_ctx_ib.rcqP, 0))
    {
      unlock_gaspi (&glb_gaspi_ctx.lockPR);
      return GASPI_ERROR;
    }

  int ne = 0;
  do
    {
      ne = ibv_poll_cq (glb_gaspi_ctx_ib.rcqP, 1, &wc_recv);
    }
  while (ne == 0);

  if ((ne < 0) || (wc_recv.status != IBV_WC_SUCCESS))
    {
      glb_gaspi_ctx.qp_state_vec[GASPI_PASSIVE_QP][wc_recv.wr_id] = 1;
      unlock_gaspi (&glb_gaspi_ctx.lockPR);
      return GASPI_ERROR;
    }

  *rem_rank = 0xffff;
  for (i = 0; i < glb_gaspi_ctx.tnc; i++)
    {
      if (glb_gaspi_ctx_ib.qpP[i]->qp_num == wc_recv.qp_num)
	{
	  *rem_rank = i;
	  break;
	}
    }


  unlock_gaspi (&glb_gaspi_ctx.lockPR);
  return GASPI_SUCCESS;

}
