/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2014

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
#include "GPI2.h"
#include "GPI2_TCP.h"

#include "tcp_device.h"

/* Communication functions */
gaspi_return_t
pgaspi_dev_write (const gaspi_segment_id_t segment_id_local,
		  const gaspi_offset_t offset_local, const gaspi_rank_t rank,
		  const gaspi_segment_id_t segment_id_remote,
		  const gaspi_offset_t offset_remote, const gaspi_size_t size,
		  const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{
  tcp_dev_wr_t wr =
    {
      .wr_id       = rank,
      .cq_handle   = glb_gaspi_ctx_tcp.scqC[queue]->num,
      .source      = glb_gaspi_ctx.rank,
      .target      = rank,
      .local_addr  = (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET + offset_local),
      .remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + NOTIFY_OFFSET + offset_remote),
      .length      = size,
      .swap        = 0,
      .opcode      = POST_RDMA_WRITE
    } ;
  
  if( write(glb_gaspi_ctx_tcp.qs_handle, &wr, sizeof(tcp_dev_wr_t)) < (ssize_t) sizeof(tcp_dev_wr_t) )
    {
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read (const gaspi_segment_id_t segment_id_local,
		 const gaspi_offset_t offset_local, const gaspi_rank_t rank,
		 const gaspi_segment_id_t segment_id_remote,
		 const gaspi_offset_t offset_remote, const gaspi_size_t size,
		 const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{
  tcp_dev_wr_t wr =
    {
      .wr_id       = rank,
      .cq_handle   = glb_gaspi_ctx_tcp.scqC[queue]->num,
      .source      = glb_gaspi_ctx.rank,
      .target       = rank,
      .local_addr  = (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET + offset_local),
      .remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + NOTIFY_OFFSET + offset_remote),
      .length      = size,
      .swap        = 0,
      .opcode      = POST_RDMA_READ
    } ;
  
  if( write(glb_gaspi_ctx_tcp.qs_handle, &wr, sizeof(tcp_dev_wr_t)) < (ssize_t) sizeof(tcp_dev_wr_t) )
    {
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}
  
gaspi_return_t
pgaspi_dev_wait (const gaspi_queue_id_t queue, int *counter, const gaspi_timeout_t timeout_ms)
{

  int ne = 0, i;
  tcp_dev_wc_t wc;

  const int nr = *counter;
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  for (i = 0; i < nr; i++)
    {
      do
	{
	  ne = tcp_dev_return_wc (glb_gaspi_ctx_tcp.scqC[queue], &wc);
	  *counter -= ne;

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
	  gaspi_print_error("Failed request to %lu. Queue %d might be broken",
			    wc.wr_id, queue);

	  glb_gaspi_ctx.qp_state_vec[queue][wc.wr_id] = 1;

	  return GASPI_ERROR;
	}
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_notify (const gaspi_segment_id_t segment_id_remote,
		   const gaspi_rank_t rank,
		   const gaspi_notification_id_t notification_id,
		   const gaspi_notification_t notification_value,
		   const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{

  uintptr_t not_addr = (uintptr_t) glb_gaspi_ctx.nsrc.buf + notification_id * 4;
  *(gaspi_notification_t *) not_addr = notification_value;
    
  tcp_dev_wr_t wr =
    {
      .wr_id       = rank,
      .cq_handle   = glb_gaspi_ctx_tcp.scqC[queue]->num,
      .source      = glb_gaspi_ctx.rank,
      .target      = rank,
      .local_addr  = (uintptr_t) not_addr,
      .remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + notification_id * 4),
      .length      = sizeof(notification_value),
      .swap        = 0,
      .opcode      = POST_RDMA_WRITE
    } ;
  
  if( write(glb_gaspi_ctx_tcp.qs_handle, &wr, sizeof(tcp_dev_wr_t)) < (ssize_t) sizeof(tcp_dev_wr_t) )
    {
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
  
}

gaspi_return_t
pgaspi_dev_write_list (const gaspi_number_t num,
		  gaspi_segment_id_t * const segment_id_local,
		  gaspi_offset_t * const offset_local,
		  const gaspi_rank_t rank,
		  gaspi_segment_id_t * const segment_id_remote,
		  gaspi_offset_t * const offset_remote,
		  gaspi_size_t * const size, const gaspi_queue_id_t queue,
		  const gaspi_timeout_t timeout_ms)
{
  int i;

  for (i = 0; i < num; i++)
    {
      tcp_dev_wr_t wr =
	{
	  .wr_id       = rank,
	  .cq_handle   = glb_gaspi_ctx_tcp.scqC[queue]->num,
	  .source      = glb_gaspi_ctx.rank,
	  .target        = rank,
	  .local_addr  = (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET + offset_local[i]),
	  .remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].addr + NOTIFY_OFFSET + offset_remote[i]),
	  .length      = size[i],
	  .swap        = 0,
	  .opcode      = POST_RDMA_WRITE
	} ;
      
      if( write(glb_gaspi_ctx_tcp.qs_handle, &wr, sizeof(tcp_dev_wr_t)) < (ssize_t) sizeof(tcp_dev_wr_t) )
	{
	  return GASPI_ERROR;
	}
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read_list (const gaspi_number_t num,
		      gaspi_segment_id_t * const segment_id_local,
		      gaspi_offset_t * const offset_local, const gaspi_rank_t rank,
		      gaspi_segment_id_t * const segment_id_remote,
		      gaspi_offset_t * const offset_remote,
		      gaspi_size_t * const size, const gaspi_queue_id_t queue,
		      const gaspi_timeout_t timeout_ms)
{
  int i;

  for (i = 0; i < num; i++)
    {
      tcp_dev_wr_t wr =
	{
	  .wr_id       = rank,
	  .cq_handle   = glb_gaspi_ctx_tcp.scqC[queue]->num,
	  .source      = glb_gaspi_ctx.rank,
	  .target        = rank,
	  .local_addr  = (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET + offset_local[i]),
	  .remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].addr + NOTIFY_OFFSET + offset_remote[i]),
	  .length      = size[i],
	  .swap        = 0,
	  .opcode      = POST_RDMA_READ
	} ;

      if( write(glb_gaspi_ctx_tcp.qs_handle, &wr, sizeof(tcp_dev_wr_t)) < (ssize_t) sizeof(tcp_dev_wr_t) )
	{
	  return GASPI_ERROR;
	}
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_write_notify (const gaspi_segment_id_t segment_id_local,
			 const gaspi_offset_t offset_local,
			 const gaspi_rank_t rank,
			 const gaspi_segment_id_t segment_id_remote,
			 const gaspi_offset_t offset_remote,
			 const gaspi_size_t size,
			 const gaspi_notification_id_t notification_id,
			 const gaspi_notification_t notification_value,
			 const gaspi_queue_id_t queue,
			 const gaspi_timeout_t timeout_ms)
{

  if (pgaspi_dev_write(segment_id_local, offset_local, rank,
		       segment_id_remote, offset_remote, size,
		       queue, timeout_ms) != GASPI_SUCCESS)
    {
      return GASPI_ERROR;
    }

  return pgaspi_dev_notify(segment_id_remote, rank, notification_id, notification_value, queue, timeout_ms);
}

gaspi_return_t
pgaspi_dev_write_list_notify (const gaspi_number_t num,
			      gaspi_segment_id_t * const segment_id_local,
			      gaspi_offset_t * const offset_local,
			      const gaspi_rank_t rank,
			      gaspi_segment_id_t * const segment_id_remote,
			      gaspi_offset_t * const offset_remote,
			      gaspi_size_t * const size,
			      const gaspi_segment_id_t segment_id_notification,
			      const gaspi_notification_id_t notification_id,
			      const gaspi_notification_t notification_value,
			      const gaspi_queue_id_t queue,
			      const gaspi_timeout_t timeout_ms)
{
  //TODO: check different with and without extra function calls
  if (pgaspi_dev_write_list(num, segment_id_local, offset_local, rank,
			    segment_id_remote, offset_remote, size,
			    queue, timeout_ms) != GASPI_SUCCESS)
    {
      return GASPI_ERROR;
    }

  return pgaspi_dev_notify(segment_id_notification, rank, notification_id, notification_value, queue, timeout_ms);
}
