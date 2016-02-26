/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2016

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
pgaspi_dev_passive_send (const gaspi_segment_id_t segment_id,
			 const gaspi_offset_t offset_local,
			 const gaspi_rank_t rank,
			 const gaspi_size_t size,
			 unsigned char *passive_counter,
			 const gaspi_timeout_t timeout_ms) 
{

  gaspi_cycles_t s0;

  const int byte_id = rank >> 3;
  const int bit_pos = rank - (byte_id * 8);
  const unsigned char bit_cmp = 1 << bit_pos;

  if (passive_counter[byte_id] & bit_cmp)
    goto checkL;

  tcp_dev_wr_t wr =
    {
      .wr_id       = rank,
      .cq_handle   = glb_gaspi_ctx_tcp.scqP->num,
      .source      = glb_gaspi_ctx.rank,
      .target      = rank,
      .local_addr  = (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.addr + offset_local),
      .remote_addr = 0UL,
      .length      = size,
      .swap        = 0,
      .compare_add = 0,
      .opcode      = POST_SEND
    } ;
  
  if( write(glb_gaspi_ctx_tcp.qpP->handle, &wr, sizeof(tcp_dev_wr_t)) < (ssize_t) sizeof(tcp_dev_wr_t) )
    {
      glb_gaspi_ctx.qp_state_vec[GASPI_PASSIVE_QP][rank] = GASPI_STATE_CORRUPT;
      return GASPI_ERROR;
    }

  passive_counter[byte_id] |= bit_cmp;
  
 checkL:
  s0 = gaspi_get_cycles();

  int ne = 0;
  tcp_dev_wc_t wc;

  do
    {
      ne = tcp_dev_return_wc (glb_gaspi_ctx_tcp.scqP, &wc);

      if (ne == 0)
	{
	  const gaspi_cycles_t s1 = gaspi_get_cycles ();
	  const gaspi_cycles_t tdelta = s1 - s0;

	  const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
	  if (ms > timeout_ms)
	    {
	      return GASPI_TIMEOUT;
	    }
	}
    }
  while (ne == 0);

  if ((ne < 0) || (wc.status != TCP_WC_SUCCESS))
    {
      glb_gaspi_ctx.qp_state_vec[GASPI_PASSIVE_QP][wc.wr_id] = GASPI_STATE_CORRUPT;

      return GASPI_ERROR;
    }

  passive_counter[byte_id] &= (~bit_cmp);
  
  return GASPI_SUCCESS;
}
 

gaspi_return_t
pgaspi_dev_passive_receive (const gaspi_segment_id_t segment_id_local,
			    const gaspi_offset_t offset_local,
			    gaspi_rank_t * const rem_rank, const gaspi_size_t size,
			    const gaspi_timeout_t timeout_ms)
{
  fd_set rfds;
  struct timeval tout;
  
  tcp_dev_wr_t wr =
    {
      .wr_id       = glb_gaspi_ctx.rank,
      .cq_handle   = glb_gaspi_ctx_tcp.rcqP->num,
      .source      = glb_gaspi_ctx.rank,
      .target      = 0,
      .local_addr  = (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].data.addr + offset_local),
      .remote_addr = 0UL,
      .length      = size,
      .swap        = 0,
      .compare_add = 0,
      .opcode      = POST_RECV
    } ;
  
  if( write(glb_gaspi_ctx_tcp.srqP, &wr, sizeof(tcp_dev_wr_t)) < (ssize_t) sizeof(tcp_dev_wr_t) )
    {
      glb_gaspi_ctx.qp_state_vec[GASPI_PASSIVE_QP][glb_gaspi_ctx.rank] = GASPI_STATE_CORRUPT;
      return GASPI_ERROR;
    }

  FD_ZERO(&rfds);
  FD_SET(glb_gaspi_ctx_tcp.channelP->read, &rfds);

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
    if(read(glb_gaspi_ctx_tcp.channelP->read, &buf, 1) < 0)
      {
	return GASPI_ERROR;
      }
  }

  int ne = 0;
  tcp_dev_wc_t wc;
  
  do
    {
      ne = tcp_dev_return_wc (glb_gaspi_ctx_tcp.rcqP, &wc);
    }
  while (ne == 0);

  if ((ne < 0) || (wc.status != TCP_WC_SUCCESS))
    {
      glb_gaspi_ctx.qp_state_vec[GASPI_PASSIVE_QP][wc.wr_id] = GASPI_STATE_CORRUPT;

      return GASPI_ERROR;
    }
  
  /* set sender rank */
  *rem_rank = (gaspi_rank_t) wc.sender;
	 
  return GASPI_SUCCESS;
}
