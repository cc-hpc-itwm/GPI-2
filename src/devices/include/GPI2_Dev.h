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

#ifndef _GPI2_DEV_H_
#define _GPI2_DEV_H_

#include "GASPI.h"

/* Device interface */

/////////////////// WORK SANDBOX /////////////////////////
///TODO: think about these

inline char *
pgaspi_dev_get_rrcd(int);

inline char *
pgaspi_dev_get_lrcd(int);

inline size_t
pgaspi_dev_get_sizeof_rc();

//TODO: following 2 functions should go away
gaspi_return_t
pgaspi_dev_group_register_mem (const int, const unsigned int);

gaspi_return_t
pgaspi_dev_group_deregister_mem (const int);

/* Groups */
int
pgaspi_dev_poll_groups();

int
pgaspi_dev_post_group_write(void *, int, int, void *, int);

//////////////////////////////////////////////////////////

#ifdef GPI2_CUDA
gaspi_return_t
pgaspi_dev_segment_alloc (const gaspi_segment_id_t,
			  const gaspi_size_t,
			  const gaspi_alloc_t);

gaspi_return_t
pgaspi_dev_segment_delete (const gaspi_segment_id_t);
#endif /* GPI2_CUDA */

int
pgaspi_dev_init_core();

int
pgaspi_dev_cleanup_core();

int
pgaspi_dev_register_mem(const gaspi_segment_id_t, const gaspi_size_t);

int
pgaspi_dev_unregister_mem(const gaspi_segment_id_t);

int
pgaspi_dev_connect_context(const int);

int
pgaspi_dev_disconnect_context(const int);

int
pgaspi_dev_create_endpoint(const int);

/* Device interface (GASPI routines) */
int
pgaspi_dev_queue_size(const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_write (const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_rank_t,
		  const gaspi_segment_id_t, const gaspi_offset_t, const unsigned int,
		  const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read (const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_rank_t,
		 const gaspi_segment_id_t, const gaspi_offset_t, const unsigned int,
		 const gaspi_queue_id_t);


gaspi_return_t
pgaspi_dev_wait (const gaspi_queue_id_t, int *, const gaspi_timeout_t);


gaspi_return_t
pgaspi_dev_write_list (const gaspi_number_t,
		       gaspi_segment_id_t * const,
		       gaspi_offset_t * const,
		       const gaspi_rank_t,
		       gaspi_segment_id_t * const,
		       gaspi_offset_t * const,
		       unsigned long* const,
		       const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read_list (const gaspi_number_t,
		      gaspi_segment_id_t * const,
		      gaspi_offset_t * const, const gaspi_rank_t,
		      gaspi_segment_id_t * const,
		      gaspi_offset_t * const,
		      unsigned long * const, const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_notify (const gaspi_segment_id_t,
		   const gaspi_rank_t,
		   const gaspi_notification_id_t,
		   const gaspi_notification_t,
		   const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_notify_waitsome (const gaspi_segment_id_t,
			    const gaspi_notification_id_t,
			    const gaspi_number_t,
			    gaspi_notification_id_t * const,
			    const gaspi_timeout_t);


gaspi_return_t
pgaspi_dev_notify_reset (const gaspi_segment_id_t,
			 const gaspi_notification_id_t,
			 gaspi_notification_t * const );

gaspi_return_t
pgaspi_dev_write_notify (const gaspi_segment_id_t,
			 const gaspi_offset_t,
			 const gaspi_rank_t,
			 const gaspi_segment_id_t,
			 const gaspi_offset_t,
			 const unsigned int,
			 const gaspi_notification_id_t,
			 const gaspi_notification_t,
			 const gaspi_queue_id_t);


gaspi_return_t
pgaspi_dev_write_list_notify (const gaspi_number_t,
			      gaspi_segment_id_t * const,
			      gaspi_offset_t * const,
			      const gaspi_rank_t,
			      gaspi_segment_id_t * const,
			      gaspi_offset_t * const,
			      unsigned int * const,
			      const gaspi_segment_id_t,
			      const gaspi_notification_id_t,
			      const gaspi_notification_t,
			      const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_atomic_fetch_add (const gaspi_segment_id_t,
			     const gaspi_offset_t, const gaspi_rank_t,
			     const gaspi_atomic_value_t,
			     gaspi_atomic_value_t * const);

gaspi_return_t
pgaspi_dev_atomic_compare_swap (const gaspi_segment_id_t,
				const gaspi_offset_t,
				const gaspi_rank_t,
				const gaspi_atomic_value_t,
				const gaspi_atomic_value_t,
				gaspi_atomic_value_t * const);

gaspi_return_t
pgaspi_dev_passive_send (const gaspi_segment_id_t,
			 const gaspi_offset_t,
			 const gaspi_rank_t,
			 const unsigned int,
			 unsigned char *,
			 const gaspi_timeout_t);

gaspi_return_t
pgaspi_dev_passive_receive (const gaspi_segment_id_t segment_id_local,
			    const gaspi_offset_t offset_local,
			    gaspi_rank_t * const rem_rank,
			    const unsigned int size,
			    const gaspi_timeout_t timeout_ms);

/* OPTION B */
/* #include "GPI2_IB_IO.h" */
/* #include "GPI2_IB_GRP.h" */
/* #include "GPI2_IB_ATOMIC.h" */
/* #include "GPI2_IB_SEG.h" */
/* #include "GPI2_IB_PASSIVE.h" */
/* #include "GPI2_IB_CONFIG.h" */

#endif //_GPI2_DEV_H_
