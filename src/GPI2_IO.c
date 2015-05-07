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
#include "GPI2_Dev.h"
#include "GPI2_Utility.h"

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

  *queue_size = (gaspi_number_t) glb_gaspi_ctx.ne_count_c[queue];

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
			      const gaspi_queue_id_t queue)
{

  
  if (glb_gaspi_ctx.rrmd[segment_id_local] == NULL)
    {
      gaspi_print_error("Invalid local segment %d (%s)", segment_id_local, func_name);
      return -1;
    }
  
  if (glb_gaspi_ctx.rrmd[segment_id_remote] == NULL)
    {
      gaspi_print_error("Invalid remote segment %d (%s)", segment_id_remote, func_name);
      return -1;
    }

  if( rank >= glb_gaspi_ctx.tnc)
    {
      gaspi_print_error("Invalid rank: %u (%s)", rank, func_name);
      return -1;
    }
  
  if( offset_local > glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].size
    || offset_remote > glb_gaspi_ctx.rrmd[segment_id_remote][rank].size)
    {
      gaspi_print_error("Invalid offsets: local %lu (segment %d of size %lu) remote %lu (segment %d of size %lu) (%s)",
			offset_local, segment_id_local, glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].size,
			offset_remote, segment_id_remote, glb_gaspi_ctx.rrmd[segment_id_remote][rank].size,
			func_name);
      return -1;
    }
    
  if( size < 1
      || size > GASPI_MAX_TSIZE_C
      || size > glb_gaspi_ctx.rrmd[segment_id_remote][rank].size
      || size > glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].size)
    {
      gaspi_print_error("Invalid size: %lu (%s)", size,func_name);
      return -1;
    }

  if (queue > glb_gaspi_cfg.queue_num - 1)
    {
      gaspi_print_error("Invalid queue: %d (%s)", queue, func_name);
      return -1;
    }

  return 0;
}
#endif //DEBUG

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
			queue) < 0)
    return GASPI_ERROR;
  
#endif

  gaspi_return_t eret = GASPI_ERROR;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_write(segment_id_local, offset_local, rank,
			  segment_id_remote,offset_remote, (unsigned int) size,
			  queue);

#ifdef DEBUG
  if(eret == GASPI_ERROR)
    {
      _print_func_params("gaspi_write", segment_id_local, offset_local, rank,
			 segment_id_remote, offset_remote, size,
			 queue, timeout_ms);
      gaspi_print_error("Elems in queue %u (max %u)",
			glb_gaspi_ctx.ne_count_c[queue],
			glb_gaspi_cfg.queue_depth);
    }
#endif

  glb_gaspi_ctx.ne_count_c[queue]++;
  
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
  
  return eret;
}

#pragma weak gaspi_read = pgaspi_read
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
			queue) < 0)
    return GASPI_ERROR;
#endif

  gaspi_return_t eret = GASPI_ERROR;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;
  
  eret = pgaspi_dev_read(segment_id_local, offset_local, rank,
			 segment_id_remote,offset_remote, (unsigned int) size,
			queue);

#ifdef DEBUG
  if(eret == GASPI_ERROR)
    {
      
      _print_func_params("gaspi_read", segment_id_local, offset_local, rank,
			 segment_id_remote, offset_remote, size,
			 queue, timeout_ms);
      
      gaspi_print_error("Elems in queue %u (max %u)",
			glb_gaspi_ctx.ne_count_c[queue],
			glb_gaspi_cfg.queue_depth);
    }
#endif

  glb_gaspi_ctx.ne_count_c[queue]++;
  
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return eret;
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

  gaspi_return_t eret = GASPI_ERROR;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_wait(queue, &glb_gaspi_ctx.ne_count_c[queue], timeout_ms);

  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return eret;
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
			    queue) < 0)
	return GASPI_ERROR;
    }
  
#endif
  gaspi_return_t eret = GASPI_ERROR;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;
  
  eret = pgaspi_dev_write_list(num, segment_id_local, offset_local, rank,
			       segment_id_remote, offset_remote, size,
			       queue);
#ifdef DEBUG
  if(eret == GASPI_ERROR)
    {
      gaspi_number_t n;
      for(n = 0; n < num; n++)
	{
	  _print_func_params("gaspi_write_list", segment_id_local[n], offset_local[n], rank,
			     segment_id_remote[n], offset_remote[n], size[n],
			     queue, timeout_ms);
	}
    }
#endif

  glb_gaspi_ctx.ne_count_c[queue] +=  num;
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return eret;
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
			    queue) < 0)
	return GASPI_ERROR;
    }
  
#endif

  gaspi_return_t eret;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;
  
  eret = pgaspi_dev_read_list(num, segment_id_local, offset_local, rank,
			      segment_id_remote, offset_remote, size,
			      queue);
#ifdef DEBUG
  if(eret == GASPI_ERROR)
    {
      gaspi_number_t n;
      for(n = 0; n < num; n++)
	{
	  _print_func_params("gaspi_read_list", segment_id_local[n], offset_local[n], rank,
			     segment_id_remote[n], offset_remote[n], size[n],
			     queue, timeout_ms);
	}
    }
#endif

  glb_gaspi_ctx.ne_count_c[queue] += num;
  
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return eret;
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
  if (glb_gaspi_ctx.rrmd[segment_id_remote] == NULL)
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

  if(notification_value == 0)
    {
      gaspi_print_error("Invalid notification value: should not be 0");
      return GASPI_ERROR;
    }

#endif
  gaspi_return_t eret = GASPI_ERROR;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_notify(segment_id_remote, rank,
			   notification_id, notification_value,
			   queue);

  glb_gaspi_ctx.ne_count_c[queue]++;
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return eret;
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
  if (glb_gaspi_ctx.rrmd[segment_id_local] == NULL)
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
  int loop = 1;
  gaspi_notification_id_t n;

  if(num == 0)
    return GASPI_SUCCESS;

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >=0 )
    {
      segPtr =  (volatile unsigned char*)glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].host_addr;
    }
  else
#endif

    segPtr = (volatile unsigned char *) glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr;
  
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
	      break;
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


#pragma weak gaspi_notify_reset = pgaspi_notify_reset
gaspi_return_t
pgaspi_notify_reset (const gaspi_segment_id_t segment_id_local,
		    const gaspi_notification_id_t notification_id,
		    gaspi_notification_t * const old_notification_val)
{

#ifdef DEBUG
  if (glb_gaspi_ctx.rrmd[segment_id_local] == NULL)
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
  if(glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].cudaDevId >= 0)
    segPtr =  (volatile unsigned char*)glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].host_addr;
  else
#endif
    segPtr = (volatile unsigned char *)
      glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].addr;
  
  volatile unsigned int *p = (volatile unsigned int *) segPtr;

  const unsigned int res = __sync_val_compare_and_swap (&p[notification_id], p[notification_id], 0);

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
			queue) < 0)
    return GASPI_ERROR;

  if(notification_value == 0)
    {
      gaspi_print_error("Invalid notification value: should not be 0");
      return GASPI_ERROR;
    }
#endif

  gaspi_return_t eret = GASPI_ERROR;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_write_notify(segment_id_local, offset_local, rank,
				 segment_id_remote, offset_remote, size,
				 notification_id, notification_value,
				 queue);

#ifdef DEBUG
  if(eret == GASPI_ERROR)
    {
      
      _print_func_params("gaspi_write_notify", segment_id_local, offset_local, rank,
			 segment_id_remote, offset_remote, size,
			 queue, timeout_ms);

      gaspi_print_error("notification_id %d\nnotification_value %u",
			notification_id,
			notification_value);
    }
#endif

  glb_gaspi_ctx.ne_count_c[queue] += 2;
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);

  return eret;  
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
			    queue) < 0)
	return GASPI_ERROR;
    }

  if(notification_value == 0)
    {
      gaspi_print_error("Invalid notification value: should not be 0");
      return GASPI_ERROR;
    }
  
#endif

  gaspi_return_t eret = GASPI_ERROR;

  if(lock_gaspi_tout (&glb_gaspi_ctx.lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_write_list_notify(num, segment_id_local, offset_local, rank,
				      segment_id_remote, offset_remote, (unsigned int *)size,
				      segment_id_notification, notification_id, notification_value,
				      queue);

#ifdef DEBUG
  if(eret == GASPI_ERROR)
    {
      
      for(n = 0; n < num; n++)
	{
	  _print_func_params("gaspi_write_list_notify", segment_id_local[n], offset_local[n], rank,
			     segment_id_remote[n], offset_remote[n], size[n],
			     queue, timeout_ms);
	}
      printf("notification_id %d\nnotification_value %u\n",
	     notification_id,
	     notification_value);
    }
#endif

  glb_gaspi_ctx.ne_count_c[queue] += (int) (num + 1);
  unlock_gaspi (&glb_gaspi_ctx.lockC[queue]);
  
  return eret;
}
