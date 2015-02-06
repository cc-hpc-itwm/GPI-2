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
#include "GPI2_IB.h"

#ifdef GPI2_CUDA
#include "GPI2_GPU.h"
#include <cuda.h>
#endif

extern gaspi_context glb_gaspi_ctx;
extern gaspi_config_t glb_gaspi_cfg;


/* Communication functions */
gaspi_return_t
pgaspi_dev_write (const gaspi_segment_id_t segment_id_local,
		  const gaspi_offset_t offset_local, const gaspi_rank_t rank,
		  const gaspi_segment_id_t segment_id_remote,
		  const gaspi_offset_t offset_remote, const unsigned int size,
		  const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  enum ibv_send_flags sf; 

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >= 0)
    {
      sf = IBV_SEND_SIGNALED;
      slist.addr =
	(uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		     offset_local);
    }
  else
#endif
    {
      sf = (size > MAX_INLINE_BYTES) ? IBV_SEND_SIGNALED : IBV_SEND_SIGNALED |
	IBV_SEND_INLINE;
     
      slist.addr = 
	(uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		     NOTIFY_OFFSET + offset_local);
    }
  
  slist.length = size;
  slist.lkey =  ((struct ibv_mr *)glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].mr)->lkey;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id_remote][rank].cudaDevId >= 0)
    swr.wr.rdma.remote_addr =(glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr +
			      offset_remote);
  else
#endif
    swr.wr.rdma.remote_addr =
      (glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + NOTIFY_OFFSET +
       offset_remote);

  swr.wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_remote][rank].rkey;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = sf;
  swr.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr))
    {
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;

      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read (const gaspi_segment_id_t segment_id_local,
		 const gaspi_offset_t offset_local, const gaspi_rank_t rank,
		 const gaspi_segment_id_t segment_id_remote,
		 const gaspi_offset_t offset_remote, const unsigned int size,
		 const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >= 0)
    slist.addr =
      (uintptr_t) (glb_gaspi_ctx_ib.
		   rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		   offset_local);
  else
#endif 
    slist.addr =
      (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		   NOTIFY_OFFSET + offset_local);
  slist.length = size;
  slist.lkey = ((struct ibv_mr *)glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].mr)->lkey;
  
#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id_remote][rank].cudaDevId >= 0)
    swr.wr.rdma.remote_addr =(glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr +
			      offset_remote);
  else
#endif
    
    swr.wr.rdma.remote_addr =
      (glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + NOTIFY_OFFSET +
       offset_remote);
  
  swr.wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_remote][rank].rkey;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_READ;
  swr.send_flags = IBV_SEND_SIGNALED;// | IBV_SEND_FENCE;
  swr.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr))
    {
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;

      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_wait (const gaspi_queue_id_t queue,
		 int * counter,
		 const gaspi_timeout_t timeout_ms)
{

  int ne = 0, i;
  struct ibv_wc wc;

  const int nr = *counter;//glb_gaspi_ctx_ib.ne_count_c[queue];
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  for (i = 0; i < nr; i++)
    {
      do
	{
	  ne = ibv_poll_cq (glb_gaspi_ctx_ib.scqC[queue], 1, &wc);
	  //	  glb_gaspi_ctx_ib.ne_count_c[queue] -= ne;
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


      if ((ne < 0) || (wc.status != IBV_WC_SUCCESS))
	{
	  gaspi_print_error("Failed request to %lu. Queue %d might be broken %s",
			    wc.wr_id, queue, ibv_wc_status_str(wc.status) );

	  glb_gaspi_ctx.qp_state_vec[queue][wc.wr_id] = 1;

	  return GASPI_ERROR;
	}
    }
#ifdef GPI2_CUDA 
  int j,k;
  for(k = 0;k < glb_gaspi_ctx.gpu_count; k++)
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
		       unsigned long * const size, const gaspi_queue_id_t queue)

{

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256];
  struct ibv_send_wr swr[256];
  int i;

  for (i = 0; i < num; i++)
    {
#ifdef GPI2_CUDA
      if(glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].cudaDevId >= 0)
	slist[i].addr =
	  (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local[i]]
		       [glb_gaspi_ctx.rank].addr +
		       offset_local[i]);

      else
#endif
	slist[i].addr =
	  (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local[i]]
		       [glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET +
		       offset_local[i]);
      slist[i].length = size[i];
      slist[i].lkey = ((struct ibv_mr *)glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].mr)->lkey;
#ifdef GPI2_CUDA
      if(glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].cudaDevId >= 0)
	swr[i].wr.rdma.remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].addr + offset_remote[i]);
      else
#endif
	swr[i].wr.rdma.remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].addr +
				      NOTIFY_OFFSET + offset_remote[i]);
      swr[i].wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].rkey;
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
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;

      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read_list (const gaspi_number_t num,
		      gaspi_segment_id_t * const segment_id_local,
		      gaspi_offset_t * const offset_local, const gaspi_rank_t rank,
		      gaspi_segment_id_t * const segment_id_remote,
		      gaspi_offset_t * const offset_remote,
		      unsigned long * const size, const gaspi_queue_id_t queue)

{

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256];
  struct ibv_send_wr swr[256];
  gaspi_number_t i;

  for (i = 0; i < num; i++)
    {
#ifdef GPI2_CUDA
      if(glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].cudaDevId >= 0)
	slist[i].addr =
	  (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].addr +
		       offset_local[i]);
      
      else
#endif
	slist[i].addr =
	  (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET +
		       offset_local[i]);
      slist[i].length = size[i];
      slist[i].lkey = ((struct ibv_mr *) glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].mr)->lkey;
      
#ifdef GPI2_CUDA
      if(glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].cudaDevId >= 0)
	swr[i].wr.rdma.remote_addr =
	  (glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].addr +
	   offset_remote[i]);
      else
#endif 
	swr[i].wr.rdma.remote_addr =
	  (glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].addr +
	   NOTIFY_OFFSET + offset_remote[i]);
      swr[i].wr.rdma.rkey =
	glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].rkey;
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
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;
      
      return GASPI_ERROR;
    }

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

  slistN.addr = (uintptr_t) (glb_gaspi_ctx.nsrc.buf + notification_id * sizeof(gaspi_notification_t));

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = 4;
  slistN.lkey = ((struct ibv_mr *)glb_gaspi_group_ctx[0].mr)->lkey;

#ifdef GPI2_CUDA
  if( glb_gaspi_ctx.rrmd[segment_id_remote][rank].cudaDevId >= 0)
    {
      swrN.wr.rdma.remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote][rank].host_addr+notification_id*4);
      swrN.wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_remote][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + notification_id * sizeof(gaspi_notification_t)); 
      swrN.wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_remote][rank].rkey;
    }
  
  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;;
  swrN.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swrN, &bad_wr))
    {
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;

      return GASPI_ERROR;
    }

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

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >= 0)
    slist.addr = (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
			      offset_local);
  else
#endif
    slist.addr = (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
			      NOTIFY_OFFSET + offset_local);

  slist.length = size;
  slist.lkey = ((struct ibv_mr *)glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].mr)->lkey;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id_remote][rank].cudaDevId >= 0)
    swr.wr.rdma.remote_addr =(glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + offset_remote);
  else
#endif
    swr.wr.rdma.remote_addr = (glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + NOTIFY_OFFSET +
			       offset_remote);

  swr.wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_remote][rank].rkey;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr = (uintptr_t) (glb_gaspi_ctx.nsrc.buf + notification_id * sizeof(gaspi_notification_t));

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = sizeof(gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *)glb_gaspi_group_ctx[0].mr)->lkey;

#ifdef GPI2_CUDA
  if((glb_gaspi_ctx.rrmd[segment_id_remote][rank].cudaDevId >= 0))
    {
      swrN.wr.rdma.remote_addr =
	(glb_gaspi_ctx.rrmd[segment_id_remote][rank].host_addr + notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_remote][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr =
	(glb_gaspi_ctx.rrmd[segment_id_remote][rank].addr + notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_remote][rank].rkey;
    }

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;;
  swrN.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr))
    {
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;

      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_write_list_notify (const gaspi_number_t num,
			      gaspi_segment_id_t * const segment_id_local,
			      gaspi_offset_t * const offset_local,
			      const gaspi_rank_t rank,
			      gaspi_segment_id_t * const segment_id_remote,
			      gaspi_offset_t * const offset_remote,
			      unsigned int * const size,
			      const gaspi_segment_id_t segment_id_notification,
			      const gaspi_notification_id_t notification_id,
			      const gaspi_notification_t notification_value,
			      const gaspi_queue_id_t queue)

{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256], slistN;
  struct ibv_send_wr swr[256], swrN;
  gaspi_number_t i;

  for (i = 0; i < num; i++)
    {

#ifdef GPI2_CUDA
      if(glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].cudaDevId >= 0)
	slist[i].addr =
	  (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local[i]]
		       [glb_gaspi_ctx.rank].addr +
		       offset_local[i]);
      
      else
#endif
	slist[i].addr =
	  (uintptr_t) (glb_gaspi_ctx.rrmd[segment_id_local[i]]
		       [glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET +
		       offset_local[i]);

      slist[i].length = size[i];
      slist[i].lkey = ((struct ibv_mr *) glb_gaspi_ctx.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].mr)->lkey;

#ifdef GPI2_CUDA
      if(glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].cudaDevId>=0)
	swr[i].wr.rdma.remote_addr =
	  (glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].addr +
	   offset_remote[i]);
      else
#endif
	swr[i].wr.rdma.remote_addr =
	  (glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].addr +
	   NOTIFY_OFFSET + offset_remote[i]);

      swr[i].wr.rdma.rkey =
	glb_gaspi_ctx.rrmd[segment_id_remote[i]][rank].rkey;
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

  slistN.addr = (uintptr_t) (glb_gaspi_ctx.nsrc.buf + notification_id * sizeof(gaspi_notification_t));

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = sizeof(gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *)glb_gaspi_group_ctx[0].mr)->lkey;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id_notification][rank].cudaDevId >= 0)
    {
      swrN.wr.rdma.remote_addr = (glb_gaspi_ctx.rrmd[segment_id_notification][rank].host_addr+notification_id*4);
      swrN.wr.rdma.rkey = glb_gaspi_ctx.rrmd[segment_id_notification][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr =
	(glb_gaspi_ctx.rrmd[segment_id_notification][rank].addr +
	 notification_id * sizeof(gaspi_notification_t));
      swrN.wr.rdma.rkey =
	glb_gaspi_ctx.rrmd[segment_id_notification][rank].rkey;
    }
  
  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;;
  swrN.next = NULL;
  
  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr[0], &bad_wr))
    {
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;
      
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}
