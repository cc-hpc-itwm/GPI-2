/*
  Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2018

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

#include "PGASPI.h"
#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_Utility.h"

#include "GPI2_SN.h"

/* Queue utilities and IO limits */
#pragma weak gaspi_queue_size = pgaspi_queue_size
gaspi_return_t
pgaspi_queue_size (const gaspi_queue_id_t queue,
		   gaspi_number_t * const queue_size)
{
  gaspi_verify_queue(queue);
  gaspi_verify_null_ptr(queue_size);
  gaspi_verify_init("gaspi_queue_size");

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  *queue_size = (gaspi_number_t) gctx->ne_count_c[queue];

  return GASPI_SUCCESS;
}

#pragma weak gaspi_queue_num = pgaspi_queue_num
gaspi_return_t
pgaspi_queue_num (gaspi_number_t * const queue_num)
{
  gaspi_verify_null_ptr(queue_num);
  gaspi_verify_init("gaspi_queue_num");

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  *queue_num = gctx->num_queues;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_queue_size_max = pgaspi_queue_size_max
gaspi_return_t
pgaspi_queue_size_max (gaspi_number_t * const queue_size_max)
{
  gaspi_verify_null_ptr(queue_size_max);
  gaspi_verify_init("gaspi_queue_size_max");

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  *queue_size_max = gctx->config->queue_size_max;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_transfer_size_min = pgaspi_transfer_size_min
gaspi_return_t
pgaspi_transfer_size_min (gaspi_size_t * const transfer_size_min)
{
  gaspi_verify_null_ptr(transfer_size_min);
  gaspi_verify_init("gaspi_transfer_size_min");

  *transfer_size_min = GASPI_MIN_TSIZE_C;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_transfer_size_max = pgaspi_transfer_size_max
gaspi_return_t
pgaspi_transfer_size_max (gaspi_size_t * const transfer_size_max)
{
  gaspi_verify_null_ptr(transfer_size_max);
  gaspi_verify_init("gaspi_transfer_size_max");

  *transfer_size_max = GASPI_MAX_TSIZE_C;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_notification_num = pgaspi_notification_num
gaspi_return_t
pgaspi_notification_num (gaspi_number_t * const notification_num)
{
  gaspi_verify_null_ptr(notification_num);
  gaspi_verify_init("gaspi_notification_num");

  *notification_num = GASPI_MAX_NOTIFICATION;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_rw_list_elem_max = pgaspi_rw_list_elem_max
gaspi_return_t
pgaspi_rw_list_elem_max (gaspi_number_t * const elem_max)
{
  gaspi_verify_null_ptr(elem_max);
  gaspi_verify_init("gaspi_rw_list_elem_max");

  *elem_max = ((1 << 8) - 1);
  return GASPI_SUCCESS;
}

#pragma weak gaspi_queue_max = pgaspi_queue_max
gaspi_return_t
pgaspi_queue_max(gaspi_number_t * const queue_max)
{
  gaspi_verify_null_ptr(queue_max);
  gaspi_verify_init("gaspi_queue_max");

  *queue_max = GASPI_MAX_QP;

  return GASPI_SUCCESS;
}

/* Queue creation/deletion/purging */
#pragma weak gaspi_queue_create = pgaspi_queue_create
gaspi_return_t
pgaspi_queue_create(gaspi_queue_id_t * const queue_id, const gaspi_timeout_t timeout_ms)
{
  gaspi_rank_t n;
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_null_ptr(queue_id);

  if( GASPI_MAX_QP == gctx->num_queues )
    return GASPI_ERROR; /* TODO: proper error code */

  if( lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms) )
    return GASPI_TIMEOUT;

  gaspi_number_t next_avail_q = __sync_fetch_and_add( & (gctx->num_queues), 0);

  /* Create it and advertise it to the already connected nodes */
  for( n = 0; n < gctx->tnc; n++ )
    {
      gaspi_rank_t i = (gctx->rank + n) % gctx->tnc;

      if( GASPI_ENDPOINT_CONNECTED == gctx->ep_conn[i].cstat )
	{
	  if( 0 != pgaspi_dev_comm_queue_create(gctx, next_avail_q, i) )
	    {
	      return GASPI_ERR_DEVICE;
	    }

	  gaspi_dev_exch_info_t * const exch_info = &(gctx->ep_conn[i].exch_info);

	  eret = gaspi_sn_command( GASPI_SN_QUEUE_CREATE, i, timeout_ms, exch_info );
	  if( GASPI_SUCCESS != eret )
	    {
	      if( GASPI_ERROR == eret )
		{
		  gctx->state_vec[GASPI_SN][i] = GASPI_STATE_CORRUPT;
		}

	      unlock_gaspi(&(gctx->ctx_lock));
	      return eret;
	    }
	}
    }

  for( n = 0; n < gctx->tnc; n++ )
    {
      gaspi_rank_t i = (gctx->rank + n) % gctx->tnc;

      if( GASPI_ENDPOINT_CONNECTED == gctx->ep_conn[i].cstat )
	{
	  if( pgaspi_dev_comm_queue_connect(gctx, next_avail_q, i) != 0 )
	    {
	      return GASPI_ERR_DEVICE;
	    }
	}
    }

  /* Increment queue counter */
  __sync_fetch_and_add( &(gctx->num_queues), 1);

  *queue_id = (gaspi_queue_id_t) next_avail_q;

  unlock_gaspi(&(gctx->ctx_lock));

  return GASPI_SUCCESS;
}

#pragma weak gaspi_queue_delete = pgaspi_queue_delete
gaspi_return_t
pgaspi_queue_delete(const gaspi_queue_id_t queue_id)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  lock_gaspi (&(gctx->ctx_lock));

  if( pgaspi_dev_comm_queue_delete(gctx, queue_id) != 0 )
    {
      unlock_gaspi(&(gctx->ctx_lock));
      return GASPI_ERR_DEVICE;
    }

  /* Decrement queue counter */
  __sync_fetch_and_sub( &(gctx->num_queues), 1);

  unlock_gaspi(&(gctx->ctx_lock));
  return GASPI_SUCCESS;
}

#pragma weak gaspi_queue_purge = pgaspi_queue_purge
gaspi_return_t
pgaspi_queue_purge(const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_queue_purge");
  gaspi_verify_queue(queue);
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_return_t eret = GASPI_ERROR;

  if(lock_gaspi_tout (&gctx->lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_purge(gctx, queue, timeout_ms);

  unlock_gaspi (&gctx->lockC[queue]);

  return eret;
}

/* Communication routines */
/* Parameter checking is done _ONLY_ when in debug mode (gaspi_verify_*) */
/* as well as printing function arguments in case of error with device
   call. */

#pragma weak gaspi_write = pgaspi_write
gaspi_return_t
pgaspi_write (const gaspi_segment_id_t segment_id_local,
	      const gaspi_offset_t offset_local,
	      const gaspi_rank_t rank,
	      const gaspi_segment_id_t segment_id_remote,
	      const gaspi_offset_t offset_remote,
	      const gaspi_size_t size,
	      const gaspi_queue_id_t queue,
	      const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_write");
  gaspi_verify_local_off(offset_local, segment_id_local, size);
  gaspi_verify_remote_off(offset_remote, segment_id_remote, rank, size);
  gaspi_verify_queue(queue);
  gaspi_verify_comm_size(size, segment_id_local, segment_id_remote, rank, GASPI_MIN_TSIZE_C, GASPI_MAX_TSIZE_C);

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_write(gctx,
			  segment_id_local, offset_local, rank,
			  segment_id_remote,offset_remote, size,
			  queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_WRITE, 1);
  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_BYTES_WRITE, size);

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;
}

#pragma weak gaspi_read = pgaspi_read
gaspi_return_t
pgaspi_read (const gaspi_segment_id_t segment_id_local,
	     const gaspi_offset_t offset_local,
	     const gaspi_rank_t rank,
	     const gaspi_segment_id_t segment_id_remote,
	     const gaspi_offset_t offset_remote,
	     const gaspi_size_t size,
	     const gaspi_queue_id_t queue,
	     const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_read");
  gaspi_verify_local_off(offset_local, segment_id_local, size);
  gaspi_verify_remote_off(offset_remote, segment_id_remote, rank, size);
  gaspi_verify_queue(queue);
  gaspi_verify_comm_size(size, segment_id_local, segment_id_remote, rank, GASPI_MIN_TSIZE_C, GASPI_MAX_TSIZE_C);

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_read(gctx,
			 segment_id_local, offset_local, rank,
			 segment_id_remote,offset_remote, size,
			 queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_READ, 1);
  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_BYTES_READ, size);

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;
}

#pragma weak gaspi_wait = pgaspi_wait
gaspi_return_t
pgaspi_wait (const gaspi_queue_id_t queue,
	     const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_wait");
  gaspi_verify_queue(queue);

  /* We need to start timing before the lock to include contention in
     lock when execution is multithreaded */
  GPI2_STATS_START_TIMER(GASPI_WAIT_TIMER);

  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_wait(gctx, queue, timeout_ms);

  if( eret != GASPI_SUCCESS )
    {
      goto endL;
    }

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_WAIT, 1);

 endL:
  unlock_gaspi (&gctx->lockC[queue]);

  GPI2_STATS_STOP_TIMER(GASPI_WAIT_TIMER);
  GPI2_STATS_INC_TIMER( GASPI_STATS_TIME_WAIT,
			GPI2_STATS_GET_TIMER(GASPI_WAIT_TIMER));

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
		   gaspi_size_t * const size,
		   const gaspi_queue_id_t queue,
		   const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if(num == 0)
    return GASPI_ERR_INV_NUM;

#ifdef DEBUG
  gaspi_verify_init("gaspi_write_list");
  gaspi_verify_queue(queue);

  gaspi_number_t n;
  for(n = 0; n < num; n++)
    {
      gaspi_verify_local_off(offset_local[n], segment_id_local[n], size[n]);
      gaspi_verify_remote_off(offset_remote[n], segment_id_remote[n], rank, size[n]);
      gaspi_verify_comm_size(size[n], segment_id_local[n], segment_id_remote[n], rank, GASPI_MIN_TSIZE_C, GASPI_MAX_TSIZE_C);
    }

#endif

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_write_list(gctx,
			       num, segment_id_local, offset_local, rank,
			       segment_id_remote, offset_remote, size,
			       queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;
}

#pragma weak gaspi_read_list = pgaspi_read_list
gaspi_return_t
pgaspi_read_list (const gaspi_number_t num,
		  gaspi_segment_id_t * const segment_id_local,
		  gaspi_offset_t * const offset_local,
		  const gaspi_rank_t rank,
		  gaspi_segment_id_t * const segment_id_remote,
		  gaspi_offset_t * const offset_remote,
		  gaspi_size_t * const size,
		  const gaspi_queue_id_t queue,
		  const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if(num == 0)
    return GASPI_ERR_INV_NUM;

#ifdef DEBUG
  gaspi_verify_init("gaspi_read_list");
  gaspi_verify_queue(queue);

  gaspi_number_t n;
  for( n = 0; n < num; n++ )
    {
      gaspi_verify_local_off(offset_local[n], segment_id_local[n], size[n]);
      gaspi_verify_remote_off(offset_remote[n], segment_id_remote[n], rank, size[n]);
      gaspi_verify_comm_size(size[n], segment_id_local[n], segment_id_remote[n], rank, GASPI_MIN_TSIZE_C, GASPI_MAX_TSIZE_C);
    }
#endif

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_read_list(gctx,
			      num, segment_id_local, offset_local, rank,
			      segment_id_remote, offset_remote, size,
			      queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;
}

#pragma weak gaspi_notify = pgaspi_notify
gaspi_return_t
pgaspi_notify (const gaspi_segment_id_t segment_id_remote,
	       const gaspi_rank_t rank,
	       const gaspi_notification_id_t notification_id,
	       const gaspi_notification_t notification_value,
	       const gaspi_queue_id_t queue,
	       const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_notify");
  gaspi_verify_segment(segment_id_remote);
  gaspi_verify_null_ptr(gctx->rrmd[segment_id_remote]);
  gaspi_verify_rank(rank);
  gaspi_verify_queue(queue);

  if(notification_value == 0)
    {
      gaspi_printf("Zero is not allowed as notification value.");
      return GASPI_ERR_INV_NOTIF_VAL;
    }

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_notify(gctx,
			   segment_id_remote, rank,
			   notification_id, notification_value,
			   queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
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
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_notify_waitsome");
  gaspi_verify_segment(segment_id_local);
  gaspi_verify_null_ptr(gctx->rrmd[segment_id_local]);
  gaspi_verify_null_ptr(first_id);

  /* We need to start timing before the lock to include contention in
     lock when execution is multithreaded */
  GPI2_STATS_START_TIMER(GASPI_WAITSOME_TIMER);

#ifdef DEBUG
  if( num > GASPI_MAX_NOTIFICATION )
    {
      return GASPI_ERR_INV_NUM;
    }

  if( num == 0 )
    {
      gaspi_print_warning("Waiting for 0 notifications (gaspi_notify_waitsome).");
    }

  if( notification_begin + (gaspi_notification_id_t) num > GASPI_MAX_NOTIFICATION )
    {
      return GASPI_ERR_INV_NOTIF_ID;
    }
#endif

  volatile unsigned char *segPtr;
  int loop = 1;
  gaspi_notification_id_t n;

  if(num == 0)
    return GASPI_SUCCESS;

  segPtr = (volatile unsigned char *) gctx->rrmd[segment_id_local][gctx->rank].notif_spc.addr;

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

		  GPI2_STATS_STOP_TIMER(GASPI_WAITSOME_TIMER);
		  GPI2_STATS_INC_TIMER( GASPI_STATS_TIME_WAITSOME,
					GPI2_STATS_GET_TIMER(GASPI_WAITSOME_TIMER));

		  return GASPI_SUCCESS;
		}
	    }

	  gaspi_delay ();
	}
    }
  else if (timeout_ms == GASPI_TEST)
    {
      for (n = notification_begin; n < (notification_begin + num); n++)
	{
	  if (p[n])
	    {
	      *first_id = n;
	      GPI2_STATS_STOP_TIMER(GASPI_WAITSOME_TIMER);
	      GPI2_STATS_INC_TIMER( GASPI_STATS_TIME_WAITSOME,
				    GPI2_STATS_GET_TIMER(GASPI_WAITSOME_TIMER));
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

      const float ms = (float) tdelta * gctx->cycles_to_msecs;
      if (ms > timeout_ms)
	{
	  GPI2_STATS_STOP_TIMER(GASPI_WAITSOME_TIMER);
	  GPI2_STATS_INC_TIMER( GASPI_STATS_TIME_WAITSOME,
				GPI2_STATS_GET_TIMER(GASPI_WAITSOME_TIMER));

	  return GASPI_TIMEOUT;
	}

      gaspi_delay ();
    }

  GPI2_STATS_STOP_TIMER(GASPI_WAITSOME_TIMER);
  GPI2_STATS_INC_TIMER( GASPI_STATS_TIME_WAITSOME,
			GPI2_STATS_GET_TIMER(GASPI_WAITSOME_TIMER));

  return GASPI_SUCCESS;
}

#pragma weak gaspi_notify_reset = pgaspi_notify_reset
gaspi_return_t
pgaspi_notify_reset (const gaspi_segment_id_t segment_id_local,
		     const gaspi_notification_id_t notification_id,
		     gaspi_notification_t * const old_notification_val)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_notify_reset");
  gaspi_verify_segment(segment_id_local);
  gaspi_verify_null_ptr(gctx->rrmd[segment_id_local]);

#ifdef DEBUG
  if(old_notification_val == NULL)
    {
      gaspi_print_warning("NULL pointer on parameter old_notification_val (gaspi_notify_reset).");
    }
#endif

  volatile unsigned int *notf_addr =
    (volatile unsigned int *) gctx->rrmd[segment_id_local][gctx->rank].notif_spc.buf;

  // TODO: one way to make sure people don't com to reset without
  // waitsome assert(p[notification_id] != 0);

  const volatile gaspi_notification_t res =
    __sync_val_compare_and_swap (&notf_addr[notification_id],
                                  notf_addr[notification_id],
                                 0);

  //TODO: at this point, p[notification_id] should be 0 or something
  //is wrong. And it cannot be the same as res

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
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_write_notify");
  gaspi_verify_local_off(offset_local, segment_id_local, size);
  gaspi_verify_remote_off(offset_remote, segment_id_remote, rank, size);
  gaspi_verify_queue(queue);
  gaspi_verify_comm_size(size, segment_id_local, segment_id_remote, rank, GASPI_MIN_TSIZE_C, GASPI_MAX_TSIZE_C);

  if(notification_value == 0)
    {
      gaspi_printf("Zero is not allowed as notification value.");
      return GASPI_ERR_INV_NOTIF_VAL;
    }

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_write_notify(gctx,
				 segment_id_local, offset_local, rank,
				 segment_id_remote, offset_remote, size,
				 notification_id, notification_value,
				 queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_WRITE_NOT, 1);
  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_BYTES_WRITE, size);

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
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
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if(num == 0)
    return GASPI_ERR_INV_NUM;

  if(notification_value == 0)
    {
      gaspi_printf("Zero is not allowed as notification value.");
      return GASPI_ERR_INV_NOTIF_VAL;
    }

#ifdef DEBUG
  gaspi_verify_init("gaspi_write_list_notify");
  gaspi_verify_queue(queue);

  gaspi_number_t n;
  for(n = 0; n < num; n++)
    {
      gaspi_verify_local_off(offset_local[n], segment_id_local[n], size[n]);
      gaspi_verify_remote_off(offset_remote[n], segment_id_remote[n], rank, size[n]);
      gaspi_verify_comm_size(size[n], segment_id_local[n], segment_id_remote[n], rank, GASPI_MIN_TSIZE_C, GASPI_MAX_TSIZE_C);
    }

#endif

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_write_list_notify(gctx,
				      num,
				      segment_id_local, offset_local, rank,
				      segment_id_remote, offset_remote, size,
				      segment_id_notification, notification_id, notification_value,
				      queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;
}

#pragma weak gaspi_read_notify = pgaspi_read_notify
gaspi_return_t
pgaspi_read_notify (const gaspi_segment_id_t segment_id_local,
		    const gaspi_offset_t offset_local,
		    const gaspi_rank_t rank,
		    const gaspi_segment_id_t segment_id_remote,
		    const gaspi_offset_t offset_remote,
		    const gaspi_size_t size,
		    const gaspi_notification_id_t notification_id,
		    const gaspi_queue_id_t queue,
		    const gaspi_timeout_t timeout_ms)
{

  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_read_notify");
  gaspi_verify_local_off(offset_local, segment_id_local, size);
  gaspi_verify_remote_off(offset_remote, segment_id_remote, rank, size);
  gaspi_verify_queue(queue);
  gaspi_verify_comm_size(size, segment_id_local, segment_id_remote, rank, GASPI_MIN_TSIZE_C, GASPI_MAX_TSIZE_C);

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_read_notify(gctx,
				segment_id_local, offset_local, rank,
				segment_id_remote, offset_remote, size,
				notification_id, queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_READ_NOT, 1);
  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_BYTES_READ, size);

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;
}

#pragma weak gaspi_read_list_notify = pgaspi_read_list_notify
gaspi_return_t
pgaspi_read_list_notify (const gaspi_number_t num,
			 gaspi_segment_id_t * const segment_id_local,
			 gaspi_offset_t * const offset_local,
			 const gaspi_rank_t rank,
			 gaspi_segment_id_t * const segment_id_remote,
			 gaspi_offset_t * const offset_remote,
			 gaspi_size_t * const size,
			 const gaspi_segment_id_t segment_id_notification,
			 const gaspi_notification_id_t notification_id,
			 const gaspi_queue_id_t queue,
			 const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( num == 0 )
    {
      return GASPI_ERR_INV_NUM;
    }

#ifdef DEBUG
  gaspi_verify_init("gaspi_read_list_notify");
  gaspi_verify_queue(queue);

  gaspi_number_t n;
  for(n = 0; n < num; n++)
    {
      gaspi_verify_local_off(offset_local[n], segment_id_local[n], size[n]);
      gaspi_verify_remote_off(offset_remote[n], segment_id_remote[n], rank, size[n]);
      gaspi_verify_comm_size(size[n], segment_id_local[n], segment_id_remote[n], rank, GASPI_MIN_TSIZE_C, GASPI_MAX_TSIZE_C);
    }

#endif

  gaspi_return_t eret = GASPI_ERROR;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  return eret;
	}
    }

  if( lock_gaspi_tout (&gctx->lockC[queue], timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_read_list_notify(gctx,
				     num,
				     segment_id_local, offset_local, rank,
				     segment_id_remote, offset_remote, size,
				     segment_id_notification, notification_id,
				     queue);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;
}
