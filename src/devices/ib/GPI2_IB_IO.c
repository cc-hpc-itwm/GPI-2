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

/* Queue utilities and IO limits */
#pragma weak gaspi_queue_size = pgaspi_queue_size
gaspi_return_t
pgaspi_queue_size (const gaspi_queue_id_t queue,
		  gaspi_number_t * const queue_size)
{
  if (queue >= glb_gaspi_cfg.queue_num)
    {
      gaspi_print_error("Invalid queue id provided");
      return GASPI_ERROR;
    }

  gaspi_verify_null_ptr(queue_size);

  *queue_size = glb_gaspi_ctx_ib.ne_count_c[queue];
  return GASPI_SUCCESS;
}

#pragma weak gaspi_queue_num = pgaspi_queue_num 
gaspi_return_t
pgaspi_queue_num (gaspi_number_t * const queue_num)
{
  gaspi_verify_null_ptr(queue_num);

  *queue_num = glb_gaspi_cfg.queue_num;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_queue_size_max = pgaspi_queue_size_max 
gaspi_return_t
pgaspi_queue_size_max (gaspi_number_t * const queue_size_max)
{
  gaspi_verify_null_ptr(queue_size_max);

  *queue_size_max = glb_gaspi_cfg.queue_depth;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_transfer_size_min = pgaspi_transfer_size_min 
gaspi_return_t
pgaspi_transfer_size_min (gaspi_size_t * const transfer_size_min)
{
  gaspi_verify_null_ptr(transfer_size_min);

  *transfer_size_min = 1;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_transfer_size_max = pgaspi_transfer_size_max 
gaspi_return_t
pgaspi_transfer_size_max (gaspi_size_t * const transfer_size_max)
{
  gaspi_verify_null_ptr(transfer_size_max);

  *transfer_size_max = GASPI_MAX_TSIZE_C;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_notification_num = pgaspi_notification_num
gaspi_return_t
pgaspi_notification_num (gaspi_number_t * const notification_num)
{
  gaspi_verify_null_ptr(notification_num);

  //TODO:?
  *notification_num = ((1 << 16) - 1);
  return GASPI_SUCCESS;
}

#pragma weak gaspi_rw_list_elem_max = pgaspi_rw_list_elem_max
gaspi_return_t
pgaspi_rw_list_elem_max (gaspi_number_t * const elem_max)
{
  gaspi_verify_null_ptr(elem_max);

  *elem_max = ((1 << 8) - 1);
  return GASPI_SUCCESS;
}

#ifdef DEBUG
static void _print_func_params(char *func_name, const gaspi_segment_id_t segment_id_local,
			       const gaspi_offset_t offset_local, const gaspi_rank_t rank,
			       const gaspi_segment_id_t segment_id_remote,
			       const gaspi_offset_t offset_remote, const gaspi_size_t size,
			       const gaspi_queue_id_t queue, const gaspi_timeout_t timeout)
{
  
  printf("%s: segment_id_local %d\n"
	 "offset_local %lu\n"
	 "rank %u\n"
	 "segment_id_remote %d\n"
	 "offset_remote %lu\n"
	 "size %lu\n"
	 "queue %d\n"
	 "timeout %lu\n",
	 func_name,
	 segment_id_local,
	 offset_local,
	 rank,
	 segment_id_remote,
	 offset_remote,
	 size,
	 queue,
	 timeout);
}

static int _check_func_params(char *func_name, const gaspi_segment_id_t segment_id_local,
			      const gaspi_offset_t offset_local, const gaspi_rank_t rank,
			      const gaspi_segment_id_t segment_id_remote,
			      const gaspi_offset_t offset_remote, const gaspi_size_t size,
			      const gaspi_queue_id_t queue, const gaspi_timeout_t timeout)
{

  if (glb_gaspi_ctx_ib.rrmd[segment_id_local] == NULL)
    {
      gaspi_print_error("Invalid local segment %d (%s)", segment_id_local, func_name);    
      return -1;
    }
  
  if (glb_gaspi_ctx_ib.rrmd[segment_id_remote] == NULL)
    {
      gaspi_print_error("Invalid remote segment %d (%s)", segment_id_remote, func_name);    
      return -1;
    }

  if( rank >= glb_gaspi_ctx.tnc)
    {
      gaspi_print_error("Invalid rank: %u (%s)", rank, func_name);    
      return -1;
    }
  
  if( offset_local > glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].size
      || offset_remote > glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].size
      )
    {
      gaspi_print_error("Invalid offsets: local %lu remote %lu (%s)",
			offset_local, offset_remote, func_name);    
      return -1;
    }
    
  if(   size < 1
     || size > GASPI_MAX_TSIZE_C
     || size > glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].size
     || size > glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].size)
    {
      gaspi_print_error("Invalid size: %lu (%s)", size,func_name);    
      return -1;
    }

  if (queue > glb_gaspi_cfg.queue_num - 1)
    {
      
      gaspi_print_error("Invalid queue: %d (%s)", queue, func_name);    
      return -1;
    }

  if(timeout < GASPI_TEST || timeout > GASPI_BLOCK)
    {
      gaspi_print_error("Invalid timeout: %lu", timeout);
      return -1;
    }
  
  return 0;
}
#endif //DEBUG

/* Communication functions */

#pragma weak gaspi_write = pgaspi_write
gaspi_return_t
pgaspi_write (const gaspi_segment_id_t segment_id_local,
	     const gaspi_offset_t offset_local, const gaspi_rank_t rank,
	     const gaspi_segment_id_t segment_id_remote,
	     const gaspi_offset_t offset_remote, const gaspi_size_t size,
	     const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if(_check_func_params("gaspi_write", segment_id_local, offset_local, rank,
			segment_id_remote, offset_remote, size,
			queue, timeout_ms) < 0)
    return GASPI_ERROR;
  
#endif
  
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  enum ibv_send_flags sf; 
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >= 0)
    {
      sf = IBV_SEND_SIGNALED;
      slist.addr =
	(uintptr_t) (glb_gaspi_ctx_ib.
		     rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		     offset_local);
    }
 else
#endif
   {
     sf = (size > MAX_INLINE_BYTES) ? IBV_SEND_SIGNALED : IBV_SEND_SIGNALED |
       IBV_SEND_INLINE;
     
     slist.addr =
       (uintptr_t) (glb_gaspi_ctx_ib.
		    rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		    NOTIFY_OFFSET + offset_local);
   }
  
  slist.length = size;
  slist.lkey =
    glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].mr->lkey;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].cudaDevId >= 0)
    swr.wr.rdma.remote_addr =(glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].addr +
			      offset_remote);
  else
#endif
    swr.wr.rdma.remote_addr =
      (glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].addr + NOTIFY_OFFSET +
       offset_remote);

  swr.wr.rdma.rkey = glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].rkey;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = sf;
  swr.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr))
    {
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;
      unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

#ifdef DEBUG
      _print_func_params("gaspi_write", segment_id_local, offset_local, rank,
			 segment_id_remote, offset_remote, size,
			 queue, timeout_ms);
      gaspi_print_error("Elems in queue %u (max %u)", 
			glb_gaspi_ctx_ib.ne_count_c[queue],
			glb_gaspi_cfg.queue_depth);
		   
#endif

      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_c[queue]++;
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_read         = pgaspi_read
gaspi_return_t
pgaspi_read (const gaspi_segment_id_t segment_id_local,
	    const gaspi_offset_t offset_local, const gaspi_rank_t rank,
	    const gaspi_segment_id_t segment_id_remote,
	    const gaspi_offset_t offset_remote, const gaspi_size_t size,
	    const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    return GASPI_ERROR;
  
  if(_check_func_params("gaspi_read", segment_id_local, offset_local, rank,
			segment_id_remote, offset_remote, size,
			queue, timeout_ms) < 0)
    return GASPI_ERROR;
#endif

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;

  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >= 0)
    slist.addr =
      (uintptr_t) (glb_gaspi_ctx_ib.
		   rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		   offset_local);
  else
#endif 
    slist.addr =
      (uintptr_t) (glb_gaspi_ctx_ib.
		   rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		   NOTIFY_OFFSET + offset_local);
  slist.length = size;
  slist.lkey =
    glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].mr->lkey;
  
#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].cudaDevId >= 0)
     swr.wr.rdma.remote_addr =(glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].addr +
      offset_remote);
  else
#endif
    
    swr.wr.rdma.remote_addr =
      (glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].addr + NOTIFY_OFFSET +
       offset_remote);
  
  swr.wr.rdma.rkey = glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].rkey;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_READ;
  swr.send_flags = IBV_SEND_SIGNALED;// | IBV_SEND_FENCE;
  swr.next = NULL;

  if (ibv_post_send (glb_gaspi_ctx_ib.qpC[queue][rank], &swr, &bad_wr))
    {
      glb_gaspi_ctx.qp_state_vec[queue][rank] = 1;
      unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
#ifdef DEBUG
      _print_func_params("gaspi_read", segment_id_local, offset_local, rank,
			 segment_id_remote, offset_remote, size,
			 queue, timeout_ms);
      
#endif
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_c[queue]++;
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_wait = pgaspi_wait
gaspi_return_t
pgaspi_wait (const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    return GASPI_ERROR;

  if (queue >= glb_gaspi_cfg.queue_num)
    {
      gaspi_print_error("Invalid queue: %d (gaspi_wait)", queue);    
      return GASPI_ERROR;
    }
#endif
  
  int ne = 0, i;
  struct ibv_wc wc;

  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  const int nr = glb_gaspi_ctx_ib.ne_count_c[queue];
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  for (i = 0; i < nr; i++)
    {

      do
	{
	  ne = ibv_poll_cq (glb_gaspi_ctx_ib.scqC[queue], 1, &wc);
	  glb_gaspi_ctx_ib.ne_count_c[queue] -= ne;

	  if (ne == 0)
	    {
	      const gaspi_cycles_t s1 = gaspi_get_cycles ();
	      const gaspi_cycles_t tdelta = s1 - s0;

	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
	      if (ms > timeout_ms)
		{
		  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
		  return GASPI_TIMEOUT;
		}
	    }

	}
      while (ne == 0);


      if ((ne < 0) || (wc.status != IBV_WC_SUCCESS))
	{
	  gaspi_print_error("Failed request to %lu. Queue %d might be broken",
			    wc.wr_id, queue);

	  glb_gaspi_ctx.qp_state_vec[queue][wc.wr_id] = 1;
	  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
	  return GASPI_ERROR;
	}

    }//for
#ifdef GPI2_CUDA 
     int j,k;
   for(k=0;k<glb_gaspi_ctx.gpu_count;k++)
     for(j = 0; j < GASPI_CUDA_EVENTS; j++)
       gpus[k].events[queue][j].ib_use=0;
#endif

  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_write_list = pgaspi_write_list
gaspi_return_t
pgaspi_write_list (const gaspi_number_t num,
		  gaspi_segment_id_t * const segment_id_local,
		  gaspi_offset_t * const offset_local,
		  const gaspi_rank_t rank,
		  gaspi_segment_id_t * const segment_id_remote,
		  gaspi_offset_t * const offset_remote,
		  gaspi_size_t * const size, const gaspi_queue_id_t queue,
		  const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  gaspi_number_t n;
  
  if (!glb_gaspi_init)
    return GASPI_ERROR;

  if(num == 0)
    {      
      gaspi_print_error("List with 0 elems");
      return GASPI_ERROR;
    }
  
  for(n = 0; n < num; n++)
    {
      if(_check_func_params("gaspi_write_list", segment_id_local[n], offset_local[n], rank,
			    segment_id_remote[n], offset_remote[n], size[n],
			    queue, timeout_ms) < 0)
	return GASPI_ERROR;
    }
  
#endif
  
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256];
  struct ibv_send_wr swr[256];
  int i;

  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  for (i = 0; i < num; i++)
    {
#ifdef GPI2_CUDA
     if(glb_gaspi_ctx_ib.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].cudaDevId >= 0)
         slist[i].addr =
        (uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local[i]]
                     [glb_gaspi_ctx.rank].addr +
                     offset_local[i]);

     else
#endif
      slist[i].addr =
	(uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local[i]]
		     [glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET +
		     offset_local[i]);
      slist[i].length = size[i];
      slist[i].lkey =
	glb_gaspi_ctx_ib.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].
	mr->lkey;
#ifdef GPI2_CUDA
     if(glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].cudaDevId >= 0)
      swr[i].wr.rdma.remote_addr =
        (glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].addr +
          offset_remote[i]);
     else
#endif
      swr[i].wr.rdma.remote_addr =
	(glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].addr +
	 NOTIFY_OFFSET + offset_remote[i]);
      swr[i].wr.rdma.rkey =
	glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].rkey;
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
      unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
#ifdef DEBUG
      int n;
      for(n = 0; n < num; n++)
	{
	  _print_func_params("gaspi_write_list", segment_id_local[n], offset_local[n], rank,
			     segment_id_remote[n], offset_remote[n], size[n],
			     queue, timeout_ms);
    }

#endif
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_c[queue] += num;
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
  return GASPI_SUCCESS;

}

#pragma weak gaspi_read_list = pgaspi_read_list
gaspi_return_t
pgaspi_read_list (const gaspi_number_t num,
		 gaspi_segment_id_t * const segment_id_local,
		 gaspi_offset_t * const offset_local, const gaspi_rank_t rank,
		 gaspi_segment_id_t * const segment_id_remote,
		 gaspi_offset_t * const offset_remote,
		 gaspi_size_t * const size, const gaspi_queue_id_t queue,
		 const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  gaspi_number_t n;
  
  if (!glb_gaspi_init)
    return GASPI_ERROR;

  if(num == 0)
    {
      gaspi_print_error("gaspi_read_list with 0 elems");
      return GASPI_ERROR;
    }
  
  
  for(n = 0; n < num; n++)
    {
      if(_check_func_params("gaspi_read_list", segment_id_local[n], offset_local[n], rank,
			    segment_id_remote[n], offset_remote[n], size[n],
			    queue, timeout_ms) < 0)
	return GASPI_ERROR;
    }
  
#endif

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256];
  struct ibv_send_wr swr[256];
  int i;

  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  for (i = 0; i < num; i++)
    {
#ifdef GPI2_CUDA
     if(glb_gaspi_ctx_ib.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].cudaDevId >= 0)
         slist[i].addr =
        (uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local[i]]
                     [glb_gaspi_ctx.rank].addr +
                     offset_local[i]);

     else
#endif
      slist[i].addr =
	(uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local[i]]
		     [glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET +
		     offset_local[i]);
      slist[i].length = size[i];
      slist[i].lkey =
	glb_gaspi_ctx_ib.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].
	mr->lkey;
#ifdef GPI2_CUDA
     if(glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].cudaDevId >= 0)
      swr[i].wr.rdma.remote_addr =
        (glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].addr +
          offset_remote[i]);
     else
#endif 
     swr[i].wr.rdma.remote_addr =
	(glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].addr +
	 NOTIFY_OFFSET + offset_remote[i]);
      swr[i].wr.rdma.rkey =
	glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].rkey;
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
      unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

#ifdef DEBUG
      int n;
      for(n = 0; n < num; n++)
	{
	  _print_func_params("gaspi_read_list", segment_id_local[n], offset_local[n], rank,
			     segment_id_remote[n], offset_remote[n], size[n],
			     queue, timeout_ms);
	}
#endif
       return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_c[queue] += num;
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
  return GASPI_SUCCESS;

}

#pragma weak gaspi_notify       = pgaspi_notify
gaspi_return_t
pgaspi_notify (const gaspi_segment_id_t segment_id_remote,
	      const gaspi_rank_t rank,
	      const gaspi_notification_id_t notification_id,
	      const gaspi_notification_t notification_value,
	      const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (glb_gaspi_ctx_ib.rrmd[segment_id_remote] == NULL)
    {
      gaspi_print_error("Invalid remote segment: %u (gaspi_notify)", segment_id_remote);    
      return GASPI_ERROR;
    }
  
  if( rank >= glb_gaspi_ctx.tnc)
    {
      gaspi_print_error("Invalid rank: %u (gaspi_notify)", rank);    
      return GASPI_ERROR;
    }

  if (queue >= glb_gaspi_cfg.queue_num)
    {
      gaspi_print_error("Invalid queue: %d (gaspi_notify)", queue);    
      return GASPI_ERROR;
    } 
#endif
  
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slistN;
  struct ibv_send_wr swrN;

  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  slistN.addr = (uintptr_t) (glb_gaspi_ctx_ib.nsrc.buf + notification_id * 4);

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = 4;
  slistN.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[0].mr)->lkey;

#ifdef GPI2_CUDA
  if( glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].cudaDevId >= 0)
    {
      swrN.wr.rdma.remote_addr = (glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].host_addr+notification_id*4);
      swrN.wr.rdma.rkey = glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr =
	(glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].addr +
	 notification_id * 4);
      swrN.wr.rdma.rkey = glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].rkey;
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
      unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_c[queue]++;
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
  return GASPI_SUCCESS;

}

#pragma weak gaspi_notify_waitsome  = pgaspi_notify_waitsome
gaspi_return_t
pgaspi_notify_waitsome (const gaspi_segment_id_t segment_id_local,
		       const gaspi_notification_id_t notification_begin,
		       const gaspi_number_t num,
		       gaspi_notification_id_t * const first_id,
		       const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (glb_gaspi_ctx_ib.rrmd[segment_id_local] == NULL)
    {
      gaspi_print_error("Invalid segment: %u  (gaspi_notify_waitsome)", segment_id_local);    
      return GASPI_ERROR;
    }
  
  if( num >= GASPI_MAX_NOTIFICATION)
    {
      gaspi_print_error("Waiting for invalid notifications number: %u  (gaspi_notify_waitsome)", num);    
      return GASPI_ERROR;
    }

  if(first_id == NULL)
    {
      gaspi_print_error("Invalid pointer on parameter first_id (gaspi_notify_waitsome)");    
      return GASPI_ERROR;
    }
  
#endif
  volatile unsigned char *segPtr;
  int n, loop = 1;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >=0 )
    {
      segPtr =  (volatile unsigned char*)glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].host_addr;
      
    }
  else
#endif
    segPtr =
      (volatile unsigned char *)
      glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr;

  volatile unsigned int *p = (volatile unsigned int *) segPtr;

  if (timeout_ms == GASPI_BLOCK)
    {

      while (loop)
	{
	  for (n = notification_begin; n < (notification_begin + num); n++)
	    {
	      if (p[n])
		{
		  *first_id = n;
		  loop = 0;
		  return GASPI_SUCCESS;
		}
	    }

	  gaspi_delay ();
	}

      return GASPI_SUCCESS;
    }
  else if (timeout_ms == GASPI_TEST)
    {

      for (n = notification_begin; n < (notification_begin + num); n++)
	{
	  if (p[n])
	    {
	      *first_id = n;
	      loop = 0;
	      return GASPI_SUCCESS;
	    }
	}

      return GASPI_TIMEOUT;
    }

  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  while (loop)
    {
      for (n = notification_begin; n < (notification_begin + num); n++)
	{
	  if (p[n])
	    {
	      *first_id = n;
	      loop = 0;
	      return GASPI_SUCCESS;
	    }
	}

      const gaspi_cycles_t s1 = gaspi_get_cycles ();
      const gaspi_cycles_t tdelta = s1 - s0;

      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      if (ms > timeout_ms)
	{
	  return GASPI_TIMEOUT;
	}

      gaspi_delay ();
    }

  return GASPI_SUCCESS;

}

#pragma weak gaspi_notify_reset     = pgaspi_notify_reset
gaspi_return_t
pgaspi_notify_reset (const gaspi_segment_id_t segment_id_local,
		    const gaspi_notification_id_t notification_id,
		    gaspi_notification_t * const old_notification_val)
{
#ifdef DEBUG
  if (glb_gaspi_ctx_ib.rrmd[segment_id_local] == NULL)
    {
      gaspi_print_error("Invalid segment: %u (gaspi_notify_reset)", segment_id_local);    
      return GASPI_ERROR;
    }
  
  if(old_notification_val == NULL)
    {
      printf("Warning: NULL pointer on parameter old_notification_val (gaspi_notify_reset)\n");    
    }
#endif

  
  volatile unsigned char *segPtr;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >= 0)
    segPtr =  (volatile unsigned char*)glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].host_addr;
  else
#endif
    segPtr = (volatile unsigned char *)
      glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr;
  
  volatile unsigned int *p = (volatile unsigned int *) segPtr;

  const unsigned int res =
    __sync_val_compare_and_swap (&p[notification_id], p[notification_id], 0);

  if(old_notification_val != NULL)
    *old_notification_val = res;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_write_notify = pgaspi_write_notify
gaspi_return_t
pgaspi_write_notify (const gaspi_segment_id_t segment_id_local,
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

#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if(_check_func_params("gaspi_write_notify", segment_id_local, offset_local, rank,
			segment_id_remote, offset_remote, size,
			queue, timeout_ms) < 0)
    return GASPI_ERROR;
  
#endif

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;

  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >= 0)
    slist.addr =
      (uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		   offset_local);
  else
#endif
    slist.addr =
      (uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr +
		   NOTIFY_OFFSET + offset_local);

  slist.length = size;
  slist.lkey = glb_gaspi_ctx_ib.rrmd[segment_id_local][glb_gaspi_ctx.rank].mr->lkey;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].cudaDevId >= 0)
    swr.wr.rdma.remote_addr =(glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].addr +
			      offset_remote);
  else
#endif
    swr.wr.rdma.remote_addr =
      (glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].addr + NOTIFY_OFFSET +
       offset_remote);

  swr.wr.rdma.rkey = glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].rkey;
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr = (uintptr_t) (glb_gaspi_ctx_ib.nsrc.buf + notification_id * 4);

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = 4; //TODO:?
  slistN.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[0].mr)->lkey;

#ifdef GPI2_CUDA
  if((glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].cudaDevId >= 0))
  {
    swrN.wr.rdma.remote_addr = (glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].host_addr+notification_id*4);
    swrN.wr.rdma.rkey = glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].host_rkey;
  }
  else
#endif
  {
    swrN.wr.rdma.remote_addr =
      (glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].addr +
       notification_id * 4);
    swrN.wr.rdma.rkey = glb_gaspi_ctx_ib.rrmd[segment_id_remote][rank].rkey;
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
    unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

#ifdef DEBUG
    _print_func_params("gaspi_write_notify", segment_id_local, offset_local, rank,
        segment_id_remote, offset_remote, size,
        queue, timeout_ms);
    gaspi_print_error("notification_id %d\nnotification_value %u",
        notification_id,
        notification_value);
#endif

    return GASPI_ERROR;
  }

  glb_gaspi_ctx_ib.ne_count_c[queue] += 2;

  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
  return GASPI_SUCCESS;
}

#pragma weak gaspi_write_list_notify = pgaspi_write_list_notify
gaspi_return_t
pgaspi_write_list_notify (const gaspi_number_t num,
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

#ifdef DEBUG
  gaspi_number_t n;
  
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if(num == 0)
    {
      gaspi_print_error("gaspi_write_list_notify with 0 elems");
      return GASPI_ERROR;
    }
  
  for(n = 0; n < num; n++)
    {
      if(_check_func_params("gaspi_write_list_notify", segment_id_local[n], offset_local[n], rank,
			    segment_id_remote[n], offset_remote[n], size[n],
			    queue, timeout_ms) < 0)
	return GASPI_ERROR;
    }
  
#endif

  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[256], slistN;
  struct ibv_send_wr swr[256], swrN;
  int i;

  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  for (i = 0; i < num; i++)
    {

#ifdef GPI2_CUDA
      if(glb_gaspi_ctx_ib.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].cudaDevId >= 0)
	slist[i].addr =
	  (uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local[i]]
		       [glb_gaspi_ctx.rank].addr +
		       offset_local[i]);
      
      else
#endif
	slist[i].addr =
	  (uintptr_t) (glb_gaspi_ctx_ib.rrmd[segment_id_local[i]]
		       [glb_gaspi_ctx.rank].addr + NOTIFY_OFFSET +
		       offset_local[i]);

      slist[i].length = size[i];
      slist[i].lkey =
	glb_gaspi_ctx_ib.rrmd[segment_id_local[i]][glb_gaspi_ctx.rank].
	mr->lkey;

#ifdef GPI2_CUDA
      if(glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].cudaDevId>=0)
	swr[i].wr.rdma.remote_addr =
	  (glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].addr +
	   offset_remote[i]);
      else
#endif
	swr[i].wr.rdma.remote_addr =
	  (glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].addr +
	   NOTIFY_OFFSET + offset_remote[i]);

      swr[i].wr.rdma.rkey =
	glb_gaspi_ctx_ib.rrmd[segment_id_remote[i]][rank].rkey;
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

  slistN.addr = (uintptr_t) (glb_gaspi_ctx_ib.nsrc.buf + notification_id * 4);

  *((unsigned int *) slistN.addr) = notification_value;

  slistN.length = 4;
  slistN.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[0].mr)->lkey;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id_notification][rank].cudaDevId >= 0)
    {
      swrN.wr.rdma.remote_addr = (glb_gaspi_ctx_ib.rrmd[segment_id_notification][rank].host_addr+notification_id*4);
      swrN.wr.rdma.rkey = glb_gaspi_ctx_ib.rrmd[segment_id_notification][rank].host_rkey;
    }
  else
#endif
    {
      swrN.wr.rdma.remote_addr =
	(glb_gaspi_ctx_ib.rrmd[segment_id_notification][rank].addr +
	 notification_id * 4);
      swrN.wr.rdma.rkey =
	glb_gaspi_ctx_ib.rrmd[segment_id_notification][rank].rkey;
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
      unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
      
#ifdef DEBUG
      for(n = 0; n < num; n++)
	{
      _print_func_params("gaspi_write_list_notify", segment_id_local[n], offset_local[n], rank,
			 segment_id_remote[n], offset_remote[n], size[n],
			 queue, timeout_ms);
	}
      printf("notification_id %d\nnotification_value %u\n",
	     notification_id,
	     notification_value);
      
#endif
      
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_c[queue] += (num + 1);
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return GASPI_SUCCESS;
}
