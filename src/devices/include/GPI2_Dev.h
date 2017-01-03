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

#ifndef _GPI2_DEV_H_
#define _GPI2_DEV_H_

#include "GASPI.h"

/* Device interface */
int
pgaspi_dev_init_core(gaspi_context_t * const gctx);

int
pgaspi_dev_cleanup_core(gaspi_context_t * const gctx);

int
pgaspi_dev_register_mem(gaspi_context_t const * const gctx, gaspi_rc_mseg_t* seg);

int
pgaspi_dev_unregister_mem(gaspi_context_t const * const gctx, gaspi_rc_mseg_t* seg);

int
pgaspi_dev_connect_context(gaspi_context_t const * const gctx, const int);

int
pgaspi_dev_disconnect_context(gaspi_context_t * const gctx, const int);

int
pgaspi_dev_create_endpoint(gaspi_context_t const * const gctx, const int i,
			   void** info, void** remote_info, size_t* info_size);

int
pgaspi_dev_comm_queue_delete(gaspi_context_t const * const gctx, const unsigned int id);

int
pgaspi_dev_comm_queue_create(gaspi_context_t const * const gctx, const unsigned int, const unsigned short);

int
pgaspi_dev_comm_queue_connect(gaspi_context_t const * const gctx, const unsigned short q, const int i);

/* Device interface (GASPI routines) */
/* Groups */
int
pgaspi_dev_poll_groups(gaspi_context_t * const gctx);

int
pgaspi_dev_post_group_write(gaspi_context_t * const gctx,
			    void *local_addr, int length,
			    int dst, void *remote_addr, unsigned char g);

int
pgaspi_dev_queue_size(const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_purge (gaspi_context_t * const gctx,
		  const gaspi_queue_id_t queue,
		  const gaspi_timeout_t timeout_ms);

gaspi_return_t
pgaspi_dev_write (gaspi_context_t * const gctx,
		  const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_rank_t,
		  const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_size_t,
		  const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read (gaspi_context_t * const gctx,
		 const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_rank_t,
		 const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_size_t,
		 const gaspi_queue_id_t);


gaspi_return_t
pgaspi_dev_wait (gaspi_context_t * const gctx, const gaspi_queue_id_t, const gaspi_timeout_t);


gaspi_return_t
pgaspi_dev_write_list (gaspi_context_t * const gctx,
		       const gaspi_number_t,
		       gaspi_segment_id_t * const,
		       gaspi_offset_t * const,
		       const gaspi_rank_t,
		       gaspi_segment_id_t * const,
		       gaspi_offset_t * const,
		       gaspi_size_t* const,
		       const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read_list (gaspi_context_t * const gctx,
		      const gaspi_number_t,
		      gaspi_segment_id_t * const,
		      gaspi_offset_t * const,
		      const gaspi_rank_t,
		      gaspi_segment_id_t * const,
		      gaspi_offset_t * const,
		      gaspi_size_t* const,
		      const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_notify (gaspi_context_t * const gctx,
		   const gaspi_segment_id_t,
		   const gaspi_rank_t,
		   const gaspi_notification_id_t,
		   const gaspi_notification_t,
		   const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_write_notify (gaspi_context_t * const gctx,
			 const gaspi_segment_id_t,
			 const gaspi_offset_t,
			 const gaspi_rank_t,
			 const gaspi_segment_id_t,
			 const gaspi_offset_t,
			 const gaspi_size_t,
			 const gaspi_notification_id_t,
			 const gaspi_notification_t,
			 const gaspi_queue_id_t);


gaspi_return_t
pgaspi_dev_write_list_notify (gaspi_context_t * const gctx,
			      const gaspi_number_t,
			      gaspi_segment_id_t * const,
			      gaspi_offset_t * const,
			      const gaspi_rank_t,
			      gaspi_segment_id_t * const,
			      gaspi_offset_t * const,
			      gaspi_size_t* const,
			      const gaspi_segment_id_t,
			      const gaspi_notification_id_t,
			      const gaspi_notification_t,
			      const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read_notify (gaspi_context_t * const gctx,
			const gaspi_segment_id_t,
			const gaspi_offset_t,
			const gaspi_rank_t,
			const gaspi_segment_id_t,
			const gaspi_offset_t,
			const gaspi_size_t,
			const gaspi_notification_id_t,
			const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read_list_notify (gaspi_context_t * const gctx,
			     const gaspi_number_t num,
			     gaspi_segment_id_t * const segment_id_local,
			     gaspi_offset_t * const offset_local,
			     const gaspi_rank_t rank,
			     gaspi_segment_id_t * const segment_id_remote,
			     gaspi_offset_t * const offset_remote,
			     gaspi_size_t * const size,
			     const gaspi_segment_id_t segment_id_notification,
			     const gaspi_notification_id_t notification_id,
			     const gaspi_queue_id_t queue);
gaspi_return_t
pgaspi_dev_atomic_fetch_add (gaspi_context_t * const gctx,
			     const gaspi_segment_id_t,
			     const gaspi_offset_t,
			     const gaspi_rank_t,
			     const gaspi_atomic_value_t);


gaspi_return_t
pgaspi_dev_atomic_compare_swap (gaspi_context_t * const gctx,
				const gaspi_segment_id_t,
				const gaspi_offset_t,
				const gaspi_rank_t,
				const gaspi_atomic_value_t,
				const gaspi_atomic_value_t);

gaspi_return_t
pgaspi_dev_passive_send (gaspi_context_t * const gctx,
			 const gaspi_segment_id_t,
			 const gaspi_offset_t,
			 const gaspi_rank_t,
			 const gaspi_size_t,
			 const gaspi_timeout_t);

gaspi_return_t
pgaspi_dev_passive_receive (gaspi_context_t * const gctx,
			    const gaspi_segment_id_t segment_id_local,
			    const gaspi_offset_t offset_local,
			    gaspi_rank_t * const rem_rank,
			    const gaspi_size_t size,
			    const gaspi_timeout_t timeout_ms);

#ifdef GPI2_CUDA
gaspi_return_t
pgaspi_dev_segment_alloc (const gaspi_segment_id_t,
			  const gaspi_size_t,
			  const gaspi_alloc_t);

gaspi_return_t
pgaspi_dev_segment_delete (const gaspi_segment_id_t);

gaspi_return_t
pgaspi_dev_gpu_write(gaspi_context_t * const gctx,
		     const gaspi_segment_id_t segment_id_local,
		     const gaspi_offset_t offset_local,
		     const gaspi_rank_t rank,
		     const gaspi_segment_id_t segment_id_remote,
		     const gaspi_offset_t offset_remote,
		     const gaspi_size_t size,
		     const gaspi_queue_id_t queue,
		     const gaspi_timeout_t timeout_ms);

gaspi_return_t
pgaspi_dev_gpu_write_notify(gaspi_context_t * const gctx,
			    const gaspi_segment_id_t segment_id_local,
			    const gaspi_offset_t offset_local,
			    const gaspi_rank_t rank,
			    const gaspi_segment_id_t segment_id_remote,
			    const gaspi_offset_t offset_remote,
			    const gaspi_size_t size,
			    const gaspi_notification_id_t notification_id,
			    const gaspi_notification_t notification_value,
			    const gaspi_queue_id_t queue,
			    const gaspi_timeout_t timeout_ms);
int
_gaspi_find_dev_numa_node(void);

#endif //GPI2_CUDA

#endif //_GPI2_DEV_H_
