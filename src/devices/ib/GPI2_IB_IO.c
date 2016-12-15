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
#include "GPI2.h"
#include "GPI2_IB.h"

#ifdef GPI2_CUDA
#include "GPI2_GPU.h"
#include <cuda.h>
#endif

extern gaspi_config_t glb_gaspi_cfg;

/* Communication functions */
gaspi_return_t
pgaspi_dev_write (const gaspi_segment_id_t segment_id_local,
		  const gaspi_offset_t offset_local,
		  const gaspi_rank_t rank,
		  const gaspi_segment_id_t segment_id_remote,
		  const gaspi_offset_t offset_remote,
		  const gaspi_size_t size,
		  const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  enum ibv_send_flags sf;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == glb_gaspi_cfg.queue_size_max )
    {
      return GASPI_QUEUE_FULL;
    }

#ifdef GPI2_CUDA
  if( gctx->rrmd[segment_id_local][gctx->rank].cuda_dev_id >= 0 )
    {
      sf = IBV_SEND_SIGNALED;
    }
  else
#endif
    {
      sf = (size > MAX_INLINE_BYTES) ? IBV_SEND_SIGNALED : IBV_SEND_SIGNALED |	IBV_SEND_INLINE;
    }

  slist.addr = (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr + offset_local);

  slist.length = size;
  slist.lkey =  ((struct ibv_mr *)gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].data.addr +  offset_remote);

  swr.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[0];
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = sf;
  swr.next = NULL;

  if( ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr) )
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read (const gaspi_segment_id_t segment_id_local,
		 const gaspi_offset_t offset_local,
		 const gaspi_rank_t rank,
		 const gaspi_segment_id_t segment_id_remote,
		 const gaspi_offset_t offset_remote,
		 const gaspi_size_t size,
		 const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == glb_gaspi_cfg.queue_size_max )
    {
      return GASPI_QUEUE_FULL;
    }

  slist.addr = (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr + offset_local);

  slist.length = size;
  slist.lkey = ((struct ibv_mr *)gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].data.addr +
			     offset_remote);

  swr.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[0];
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_READ;
  swr.send_flags = IBV_SEND_SIGNALED;// | IBV_SEND_FENCE;
  swr.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr))
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_purge (const gaspi_queue_id_t queue,
		  const gaspi_timeout_t timeout_ms)
{
  int ne = 0, i;
  struct ibv_wc wc;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  const int nr = gctx->ne_count_c[queue];
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  for (i = 0; i < nr; i++)
    {
      do
	{
	  ne = ibv_poll_cq (glb_gaspi_ctx_ib.scqC[queue], 1, &wc);
	  gctx->ne_count_c[queue] -= ne;

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
    }

#ifdef GPI2_CUDA
  int j, k;
  for(k = 0;k < gctx->gpu_count; k++)
    {
      for(j = 0; j < GASPI_CUDA_EVENTS; j++)
	gpus[k].events[queue][j].ib_use = 0;
    }
#endif

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_wait (const gaspi_queue_id_t queue,
		 const gaspi_timeout_t timeout_ms)
{

  int ne = 0, i;
  struct ibv_wc wc;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  const int nr = gctx->ne_count_c[queue];
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  for (i = 0; i < nr; i++)
    {
      do
	{
	  ne = ibv_poll_cq (glb_gaspi_ctx_ib.scqC[queue], 1, &wc);
	  gctx->ne_count_c[queue] -= ne; //TODO: this should be done below, when ne > 0

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


      if ((ne < 0) || (wc.status != IBV_WC_SUCCESS))
	{
	  //TODO: for now here because we have to identify the rank
	  // but should be out of device?
	  gctx->qp_state_vec[queue][wc.wr_id] = GASPI_STATE_CORRUPT;
	  gaspi_print_error("Failed request to %lu. Queue %d might be broken %s",
			    wc.wr_id, queue, ibv_wc_status_str(wc.status) );

	  return GASPI_ERROR;
	}
    }
#ifdef GPI2_CUDA
  int j,k;
  for(k = 0;k < gctx->gpu_count; k++)
    {
      for(j = 0; j < GASPI_CUDA_EVENTS; j++)
	gpus[k].events[queue][j].ib_use = 0;
    }

#endif

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_write_list (const gaspi_number_t num,
		       gaspi_segment_id_t * const segment_id_local,
		       gaspi_offset_t * const offset_local,
		       const gaspi_rank_t rank,
		       gaspi_segment_id_t * const segment_id_remote,
		       gaspi_offset_t * const offset_remote,
		       gaspi_size_t * const size,
		       const gaspi_queue_id_t queue)
{

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256];
  struct ibv_send_wr swr[256];
  gaspi_number_t i;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == (glb_gaspi_cfg.queue_size_max - num + 1) )
    {
      return GASPI_QUEUE_FULL;
    }


  for (i = 0; i < num; i++)
    {
      slist[i].addr = (uintptr_t) (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr + offset_local[i]);

      slist[i].length = size[i];
      slist[i].lkey = ((struct ibv_mr *)gctx->rrmd[segment_id_local[i]][gctx->rank].mr[0])->lkey;

      swr[i].wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote[i]][rank].data.addr + offset_remote[i]);

      swr[i].wr.rdma.rkey = gctx->rrmd[segment_id_remote[i]][rank].rkey[0];
      swr[i].sg_list = &slist[i];
      swr[i].num_sge = 1;
      swr[i].wr_id = rank;
      swr[i].opcode = IBV_WR_RDMA_WRITE;
      swr[i].send_flags = IBV_SEND_SIGNALED;

      if (i == (num - 1))
	swr[i].next = NULL;
      else
	swr[i].next = &swr[i + 1];
    }

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr[0], &bad_wr))
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue] +=  num;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read_list (const gaspi_number_t num,
		      gaspi_segment_id_t * const segment_id_local,
		      gaspi_offset_t * const offset_local,
		      const gaspi_rank_t rank,
		      gaspi_segment_id_t * const segment_id_remote,
		      gaspi_offset_t * const offset_remote,
		      gaspi_size_t * const size,
		      const gaspi_queue_id_t queue)

{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256];
  struct ibv_send_wr swr[256];
  gaspi_number_t i;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == (glb_gaspi_cfg.queue_size_max - num + 1) )
    {
      return GASPI_QUEUE_FULL;
    }

  for (i = 0; i < num; i++)
    {
      slist[i].addr = (uintptr_t) (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
				     offset_local[i]);

      slist[i].length = size[i];
      slist[i].lkey = ((struct ibv_mr *) gctx->rrmd[segment_id_local[i]][gctx->rank].mr[0])->lkey;

      swr[i].wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote[i]][rank].data.addr +
				    offset_remote[i]);

      swr[i].wr.rdma.rkey = gctx->rrmd[segment_id_remote[i]][rank].rkey[0];
      swr[i].sg_list = &slist[i];
      swr[i].num_sge = 1;
      swr[i].wr_id = rank;
      swr[i].opcode = IBV_WR_RDMA_READ;
      swr[i].send_flags = IBV_SEND_SIGNALED;
      if (i == (num - 1))
	swr[i].next = NULL;
      else
	swr[i].next = &swr[i + 1];
    }

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr[0], &bad_wr))
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue] += num;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_notify (const gaspi_segment_id_t segment_id_remote,
		   const gaspi_rank_t rank,
		   const gaspi_notification_id_t notification_id,
		   const gaspi_notification_t notification_value,
		   const gaspi_queue_id_t queue)
{

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slistN;
  struct ibv_send_wr swrN;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == glb_gaspi_cfg.queue_size_max )
    {
      return GASPI_QUEUE_FULL;
    }

  slistN.addr = (uintptr_t) (gctx->nsrc.notif_spc.buf + notification_id * sizeof(gaspi_notification_t));

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = sizeof(gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *) gctx->nsrc.mr[1])->lkey;

#ifdef GPI2_CUDA
  if( gctx->rrmd[segment_id_remote][rank].cuda_dev_id >= 0)
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].host_addr + notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].notif_spc.addr + notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[1];
    }

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swrN.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swrN, &bad_wr))
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_write_notify (const gaspi_segment_id_t segment_id_local,
			 const gaspi_offset_t offset_local,
			 const gaspi_rank_t rank,
			 const gaspi_segment_id_t segment_id_remote,
			 const gaspi_offset_t offset_remote,
			 const unsigned int size,
			 const gaspi_notification_id_t notification_id,
			 const gaspi_notification_t notification_value,
			 const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == glb_gaspi_cfg.queue_size_max - 1 )
    {
      return GASPI_QUEUE_FULL;
    }

  slist.addr = (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
			    offset_local);

  slist.length = size;
  slist.lkey = ((struct ibv_mr *)gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].data.addr +
			     offset_remote);

  swr.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[0];
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr = (uintptr_t) (gctx->nsrc.notif_spc.buf + notification_id * sizeof(gaspi_notification_t));

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = sizeof(gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *) gctx->nsrc.mr[1])->lkey;

#ifdef GPI2_CUDA
  if((gctx->rrmd[segment_id_remote][rank].cuda_dev_id >= 0))
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].host_addr + notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].notif_spc.addr + notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[1];
    }

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;;
  swrN.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr))
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue] += 2;

  return GASPI_SUCCESS;
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
			      const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256], slistN;
  struct ibv_send_wr swr[256], swrN;
  gaspi_number_t i;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == (glb_gaspi_cfg.queue_size_max - num - 2) )
    {
      return GASPI_QUEUE_FULL;
    }

  for (i = 0; i < num; i++)
    {

      slist[i].addr = (uintptr_t) (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
				   offset_local[i]);

      slist[i].length = size[i];
      slist[i].lkey = ((struct ibv_mr *) gctx->rrmd[segment_id_local[i]][gctx->rank].mr[0])->lkey;

      swr[i].wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote[i]][rank].data.addr +
				    offset_remote[i]);

      swr[i].wr.rdma.rkey = gctx->rrmd[segment_id_remote[i]][rank].rkey[0];
      swr[i].sg_list = &slist[i];
      swr[i].num_sge = 1;
      swr[i].wr_id = rank;
      swr[i].opcode = IBV_WR_RDMA_WRITE;
      swr[i].send_flags = IBV_SEND_SIGNALED;
      if (i == (num - 1))
	swr[i].next = &swrN;
      else
	swr[i].next = &swr[i + 1];
    }

  slistN.addr = (uintptr_t) (gctx->nsrc.notif_spc.buf + notification_id * sizeof(gaspi_notification_t));

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = sizeof(gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *) gctx->nsrc.mr[1])->lkey;

#ifdef GPI2_CUDA
  if(gctx->rrmd[segment_id_notification][rank].cuda_dev_id >= 0)
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_notification][rank].host_addr + notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_notification][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_notification][rank].notif_spc.addr +
				  notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_notification][rank].rkey[1];
    }

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swrN.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr[0], &bad_wr))
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue] += (int) (num + 1);

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read_notify (const gaspi_segment_id_t segment_id_local,
			const gaspi_offset_t offset_local,
			const gaspi_rank_t rank,
			const gaspi_segment_id_t segment_id_remote,
			const gaspi_offset_t offset_remote,
			const unsigned int size,
			const gaspi_notification_id_t notification_id,
			const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == glb_gaspi_cfg.queue_size_max - 1 )
    {
      return GASPI_QUEUE_FULL;
    }

  slist.addr = (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr + offset_local);

  slist.length = size;
  slist.lkey = ((struct ibv_mr *)gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].data.addr + offset_remote);

  swr.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[0];
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_READ;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr = (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].notif_spc.addr + notification_id * sizeof(gaspi_notification_t));
  slistN.length = sizeof(gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *)gctx->rrmd[segment_id_local][gctx->rank].mr[1])->lkey;

#ifdef GPI2_CUDA
  if((gctx->rrmd[segment_id_remote][rank].cuda_dev_id >= 0))
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].host_addr + NOTIFY_OFFSET - sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].notif_spc.addr + NOTIFY_OFFSET - sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[1];
    }

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_READ;
  swrN.send_flags = IBV_SEND_SIGNALED;// | IBV_SEND_FENCE;;
  swrN.next = NULL;

  if( ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr) )
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue] += 2;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read_list_notify (const gaspi_number_t num,
			     gaspi_segment_id_t * const segment_id_local,
			     gaspi_offset_t * const offset_local,
			     const gaspi_rank_t rank,
			     gaspi_segment_id_t * const segment_id_remote,
			     gaspi_offset_t * const offset_remote,
			     gaspi_size_t * const size,
			     const gaspi_segment_id_t segment_id_notification,
			     const gaspi_notification_id_t notification_id,
			     const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256], slistN;
  struct ibv_send_wr swr[256], swrN;
  gaspi_number_t i;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->ne_count_c[queue] == (glb_gaspi_cfg.queue_size_max - num - 2) )
    {
      return GASPI_QUEUE_FULL;
    }

  for (i = 0; i < num; i++)
    {
      slist[i].addr = (uintptr_t) (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
				     offset_local[i]);

      slist[i].length = size[i];
      slist[i].lkey = ((struct ibv_mr *) gctx->rrmd[segment_id_local[i]][gctx->rank].mr[0])->lkey;

      swr[i].wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote[i]][rank].data.addr +
				    offset_remote[i]);

      swr[i].wr.rdma.rkey = gctx->rrmd[segment_id_remote[i]][rank].rkey[0];
      swr[i].sg_list = &slist[i];
      swr[i].num_sge = 1;
      swr[i].wr_id = rank;
      swr[i].opcode = IBV_WR_RDMA_READ;
      swr[i].send_flags = IBV_SEND_SIGNALED;
      if (i == (num - 1))
	swr[i].next = &swrN;
      else
	swr[i].next = &swr[i + 1];
    }

  slistN.addr = (uintptr_t) (gctx->rrmd[segment_id_notification][gctx->rank].notif_spc.addr + notification_id * sizeof(gaspi_notification_t));
  slistN.length = sizeof(gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *)gctx->rrmd[segment_id_notification][gctx->rank].mr[1])->lkey;

#ifdef GPI2_CUDA
  if(gctx->rrmd[segment_id_notification][rank].cuda_dev_id >= 0)
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_notification][rank].host_addr + NOTIFY_OFFSET - sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_notification][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_notification][rank].notif_spc.addr + NOTIFY_OFFSET - sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_notification][rank].rkey[1];
    }

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_READ;
  swrN.send_flags = IBV_SEND_SIGNALED;// | IBV_SEND_FENCE;
  swrN.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr[0], &bad_wr))
    {
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue] += (num + 1) ;

  return GASPI_SUCCESS;
}

#ifdef GPI2_CUDA
/* TODO: maybe rename it? gaspi_post_write_from_host */
static int
_gaspi_event_send(gaspi_cuda_event_t* event, int queue)
{
  struct ibv_send_wr swr;
  struct ibv_sge slist;
  struct ibv_send_wr *bad_wr;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  swr.wr.rdma.rkey = gctx->rrmd[event->segment_remote][event->rank].rkey[0];
  swr.sg_list    = &slist;
  swr.num_sge    = 1;
  swr.wr_id      = event->rank;
  swr.opcode     = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next       = NULL;

  slist.addr = (uintptr_t) (char*)(gctx->rrmd[event->segment_local][event->rank].host_ptr + NOTIFY_OFFSET + event->offset_local);

  slist.length = event->size;
  slist.lkey = ((struct ibv_mr *)gctx->rrmd[event->segment_local][gctx->rank].host_mr)->lkey;

  swr.wr.rdma.remote_addr = (gctx->rrmd[event->segment_remote][event->rank].data.addr + event->offset_remote);

  if( ibv_post_send(glb_gaspi_ctx_ib.qpC[queue][event->rank], &swr, &bad_wr) )
    {
      //TODO:not here
      gctx->qp_state_vec[queue][event->rank] = GASPI_STATE_CORRUPT;
      return -1;
    }

  gctx->ne_count_c[queue]++;

  event->ib_use = 1;

  return 0;
}

gaspi_return_t
pgaspi_dev_gpu_write(const gaspi_segment_id_t segment_id_local,
		 const gaspi_offset_t offset_local,
		 const gaspi_rank_t rank,
		 const gaspi_segment_id_t segment_id_remote,
		 const gaspi_offset_t offset_remote,
		 const gaspi_size_t size,
		 const gaspi_queue_id_t queue,
		 const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->rrmd[segment_id_local][gctx->rank].cuda_dev_id < 0 ||
      size <= GASPI_GPU_DIRECT_MAX )
    {
      return pgaspi_dev_write(segment_id_local, offset_local, rank,
			     segment_id_remote, offset_remote, size,
			     queue);
    }

  char* host_ptr = (char*)(gctx->rrmd[segment_id_local][gctx->rank].host_ptr + NOTIFY_OFFSET + offset_local);
  char* device_ptr = (char*)(gctx->rrmd[segment_id_local][gctx->rank].data.addr + offset_local);

  //TODO: look every time for a gpu? why?
  gaspi_gpu_t* agpu = _gaspi_find_gpu(gctx->rrmd[segment_id_local][gctx->rank].cuda_dev_id);
  if( !agpu )
    {
      gaspi_print_error("No GPU found or not initialized.");
      return GASPI_ERROR;
    }

  int size_left = size;
  int copy_size = 0;
  int gpu_offset = 0;
  const int BLOCK_SIZE = GASPI_GPU_BUFFERED;

  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  while(size_left > 0)
    {
      int i;
      for(i = 0; i < GASPI_CUDA_EVENTS; i++)
	{
	  if(size_left > BLOCK_SIZE)
	    {
	      copy_size = BLOCK_SIZE;
	    }
	  else
	    {
	      copy_size = size_left;
	    }

	  /* Start copy to host */
	  if( cudaMemcpyAsync( host_ptr + gpu_offset,
			       device_ptr + gpu_offset,
			       copy_size,
			       cudaMemcpyDeviceToHost,
			       agpu->streams[queue]) )
	    {
	      return GASPI_ERROR;
	    }
	  //TODO: not here
	  gctx->ne_count_c[queue]++;

	  /* Keep track of event to query later on */
	  agpu->events[queue][i].segment_remote = segment_id_remote;
	  agpu->events[queue][i].segment_local = segment_id_local;
	  agpu->events[queue][i].size = copy_size;
	  agpu->events[queue][i].rank = rank;
	  agpu->events[queue][i].offset_local = offset_local+gpu_offset;
	  agpu->events[queue][i].offset_remote = offset_remote+gpu_offset;
	  agpu->events[queue][i].in_use = 1;

	  cudaError_t err = cudaEventRecord(agpu->events[queue][i].event, agpu->streams[queue]);
	  if( err != cudaSuccess )
	    {
	      //TODO: not here
	      gctx->qp_state_vec[queue][rank] = GASPI_STATE_CORRUPT;
	      return GASPI_ERROR;
	    }

	  gpu_offset += copy_size;
	  size_left -= copy_size;

	  if( size_left == 0 )
	    {
	      break;
	    }

	  /* We keep polling the queue to avoid overruning it */
	  if( agpu->events[queue][i].ib_use )
	    {
	      struct ibv_wc wc;
	      int ne;
	      do
		{
		  ne = ibv_poll_cq (glb_gaspi_ctx_ib.scqC[queue], 1, &wc);
		  gctx->ne_count_c[queue] -= ne; //TODO: this should be done below, when ne > 0
		  if( ne == 0 )
		    {
		      const gaspi_cycles_t s1 = gaspi_get_cycles ();
		      const gaspi_cycles_t tdelta = s1 - s0;

		      const float ms = (float) tdelta * gctx->cycles_to_msecs;
		      if (ms > timeout_ms)
			{
			  return GASPI_TIMEOUT;
			}
		    }
		} while(ne == 0);
	      //TODO: handle error case (ne < 0)
	      agpu->events[queue][i].ib_use = 0;
	    }
	}

      for(i = 0; i < GASPI_CUDA_EVENTS; i++)
	{
	  cudaError_t error;
	  if ( agpu->events[queue][i].in_use == 1 )
	    {
	      /* Wait for memcpy to finish and trigger operation (gaspi_event_send) */
	      do
		{
		  error = cudaEventQuery(agpu->events[queue][i].event );
		  if( cudaSuccess == error )
		    {
		      if( _gaspi_event_send(&agpu->events[queue][i], queue) )
			{
			  return GASPI_ERROR;
			}

		      agpu->events[queue][i].in_use = 0;
		    }
		  else if( error == cudaErrorNotReady )
		    {
		      const gaspi_cycles_t s1 = gaspi_get_cycles ();
		      const gaspi_cycles_t tdelta = s1 - s0;

		      const float ms = (float) tdelta * gctx->cycles_to_msecs;
		      if( ms > timeout_ms )
			{
			  return GASPI_TIMEOUT;
			}
		    }
		  else
		    {
		      return GASPI_ERROR;
		    }
		} while(error != cudaSuccess);
	    }
	}
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_gpu_write_notify(const gaspi_segment_id_t segment_id_local,
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
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->rrmd[segment_id_local][gctx->rank].cuda_dev_id < 0 ||
      size <= GASPI_GPU_DIRECT_MAX )
    {
      return pgaspi_dev_write_notify( segment_id_local, offset_local, rank,
				 segment_id_remote, offset_remote, size,
				 notification_id, notification_value,
				 queue);
    }

  char *host_ptr = (char*)(gctx->rrmd[segment_id_local][gctx->rank].host_ptr + NOTIFY_OFFSET + offset_local);
  char* device_ptr =(char*)(gctx->rrmd[segment_id_local][gctx->rank].data.addr + offset_local);

  //TODO: again the look up for the gpu?
  gaspi_gpu_t* agpu = _gaspi_find_gpu(gctx->rrmd[segment_id_local][gctx->rank].cuda_dev_id);
  if( !agpu )
    {
      gaspi_print_error("No GPU found or not initialized.");
      return GASPI_ERROR;
    }

  int copy_size = 0;
  int gpu_offset = 0;
  int size_left = size;
  int BLOCK_SIZE= GASPI_GPU_BUFFERED;

  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  while(size_left > 0)
    {
      int i;
      for(i = 0; i < GASPI_CUDA_EVENTS; i++)
	{
	  if(size_left > BLOCK_SIZE)
	    {
	      copy_size = BLOCK_SIZE;
	    }
	  else
	    {
	      copy_size = size_left;
	    }

	  if( cudaMemcpyAsync( host_ptr + gpu_offset,
			       device_ptr + gpu_offset,
			       copy_size,
			       cudaMemcpyDeviceToHost,
			       agpu->streams[queue]))
	    {
	      return GASPI_ERROR;
	    }
	  //TODO: not here?
	  gctx->ne_count_c[queue]++;

	  agpu->events[queue][i].segment_remote = segment_id_remote;
	  agpu->events[queue][i].segment_local = segment_id_local;
	  agpu->events[queue][i].size = copy_size;
	  agpu->events[queue][i].rank = rank;
	  agpu->events[queue][i].offset_local = offset_local+gpu_offset;
	  agpu->events[queue][i].offset_remote = offset_remote+gpu_offset;
	  agpu->events[queue][i].in_use  = 1;

	  cudaError_t err = cudaEventRecord(agpu->events[queue][i].event,agpu->streams[queue]);
	  if( err != cudaSuccess )
	    {
	      return GASPI_ERROR;
	    }

	  /* We keep polling the queue to avoid overruning it */
	  if( agpu->events[queue][i].ib_use )
	    {
	      struct ibv_wc wc;
	      int ne;
	      do
		{
		  ne = ibv_poll_cq (glb_gaspi_ctx_ib.scqC[queue], 1, &wc);
		  gctx->ne_count_c[queue] -= ne; //TODO: this should be done below, when ne > 0
		  if( ne == 0 )
		    {
		      const gaspi_cycles_t s1 = gaspi_get_cycles ();
		      const gaspi_cycles_t tdelta = s1 - s0;

		      const float ms = (float) tdelta * gctx->cycles_to_msecs;
		      if (ms > timeout_ms)
			{
			  return GASPI_TIMEOUT;
			}
		    }
		} while(ne == 0);
	      //TODO: handle error case (ne < 0)
	      agpu->events[queue][i].ib_use = 0;
	    }

	  gpu_offset += copy_size;
	  size_left -= copy_size;
	  if( size_left == 0 )
	    {
	      break;
	    }
	}

      for(i = 0; i < GASPI_CUDA_EVENTS; i++)
	{
	  cudaError_t error;
	  if (agpu->events[queue][i].in_use == 1 )
	    {
	      do
		{
		  error = cudaEventQuery(agpu->events[queue][i].event );
		  if( cudaSuccess == error )
		    {
		      if (_gaspi_event_send(&agpu->events[queue][i],queue) )
			{
			  return GASPI_ERROR;
			}

		      agpu->events[queue][i].in_use  = 0;
		    }
		  else if(error == cudaErrorNotReady)
		    {
		      const gaspi_cycles_t s1 = gaspi_get_cycles ();
		      const gaspi_cycles_t tdelta = s1 - s0;

		      const float ms = (float) tdelta * gctx->cycles_to_msecs;
		      if (ms > timeout_ms)
			{
			  return GASPI_TIMEOUT;
			}
		    }
		  else
		    {
		      return GASPI_ERROR;
		    }
		} while(error != cudaSuccess);
	    }
	}
    }

  /* Now send the notification */
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slistN;
  struct ibv_send_wr swrN;

  slistN.addr = (uintptr_t)(gctx->nsrc.notif_spc.buf + notification_id * sizeof(gaspi_notification_id_t));

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = sizeof(gaspi_notification_id_t);
  slistN.lkey =((struct ibv_mr *) gctx->nsrc.mr)->lkey;

  if( gctx->rrmd[segment_id_remote][rank].cuda_dev_id >= 0 )
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].host_addr + notification_id * sizeof(gaspi_notification_id_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].host_rkey;
    }
  else
    {
      swrN.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].notif_spc.addr + notification_id * sizeof(gaspi_notification_id_t));
      swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[1];
    }

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swrN.next = NULL;

  if( ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swrN, &bad_wr) )
    {
      gctx->qp_state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      return GASPI_ERROR;
    }

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

int
_gaspi_find_dev_numa_node(void)
{
  char path[128];
  int numa_node;
  FILE *sysfile = NULL;

  sprintf(path, "/sys/class/infiniband/%s/device/numa_node",
	  ibv_get_device_name(glb_gaspi_ctx_ib.ib_dev));

  sysfile = fopen(path, "r");
  if( sysfile == NULL )
    {
      gaspi_print_error("Failed to open %s.", path);
      return -1;
    }

  fscanf (sysfile, "%1d", &numa_node);
  fclose(sysfile);

  return numa_node;
}

#endif //GPI2_CUDA
