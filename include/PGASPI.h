/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2019

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

#ifndef PGPI2_H
#define PGPI2_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "GASPI.h"

  gaspi_return_t pgaspi_config_get (gaspi_config_t * const config);

  gaspi_return_t pgaspi_config_set (const gaspi_config_t new_config);

  gaspi_return_t pgaspi_version (float *version);

  gaspi_return_t pgaspi_proc_init (const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_initialized (gaspi_number_t * initialized);

  gaspi_return_t pgaspi_proc_term (const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_proc_local_rank (gaspi_rank_t * const local_rank);

  gaspi_return_t pgaspi_proc_local_num (gaspi_rank_t * const local_num);

  gaspi_return_t pgaspi_proc_rank (gaspi_rank_t * const rank);

  gaspi_return_t pgaspi_proc_num (gaspi_rank_t * const proc_num);

  gaspi_return_t pgaspi_proc_kill (const gaspi_rank_t rank,
				   const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_connect (const gaspi_rank_t rank,
				 const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_disconnect (const gaspi_rank_t rank,
				    const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_group_create (gaspi_group_t * const group);


  gaspi_return_t pgaspi_group_delete (const gaspi_group_t group);

  gaspi_return_t pgaspi_group_add (const gaspi_group_t group,
				   const gaspi_rank_t rank);

  gaspi_return_t pgaspi_group_commit (const gaspi_group_t group,
				      const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_group_num (gaspi_number_t * const group_num);

  gaspi_return_t pgaspi_group_size (const gaspi_group_t group,
				    gaspi_number_t * const group_size);

  gaspi_return_t pgaspi_group_ranks (const gaspi_group_t group,
				     gaspi_rank_t * const group_ranks);

  gaspi_return_t pgaspi_group_max (gaspi_number_t * const group_max);

  gaspi_return_t pgaspi_segment_alloc (const gaspi_segment_id_t segment_id,
				       const gaspi_size_t size,
				       const gaspi_alloc_t alloc_policy);

  gaspi_return_t pgaspi_segment_delete (const gaspi_segment_id_t segment_id);


  gaspi_return_t pgaspi_segment_register (const gaspi_segment_id_t segment_id,
					  const gaspi_rank_t rank,
					  const gaspi_timeout_t timeout_ms);


  gaspi_return_t pgaspi_segment_create (const gaspi_segment_id_t segment_id,
					const gaspi_size_t size,
					const gaspi_group_t group,
					const gaspi_timeout_t timeout_ms,
					const gaspi_alloc_t alloc_policy);
  gaspi_return_t pgaspi_segment_bind ( gaspi_segment_id_t const segment_id,
				       gaspi_pointer_t const pointer,
				       gaspi_size_t const size,
				       gaspi_memory_description_t const memory_description);

  gaspi_return_t pgaspi_segment_use ( gaspi_segment_id_t const segment_id,
				      gaspi_pointer_t const pointer,
				      gaspi_size_t const size,
				      gaspi_group_t const group,
				      gaspi_timeout_t const timeout,
				      gaspi_memory_description_t const memory_description);

  gaspi_return_t pgaspi_segment_num (gaspi_number_t * const segment_num);

  gaspi_return_t pgaspi_segment_list (const gaspi_number_t num,
				      gaspi_segment_id_t * const segment_id_list);

  gaspi_return_t pgaspi_segment_ptr (const gaspi_segment_id_t segment_id,
				     gaspi_pointer_t * ptr);


  gaspi_return_t pgaspi_segment_size (const gaspi_segment_id_t segment_id,
				      const gaspi_rank_t rank,
				      gaspi_size_t * const size);


  gaspi_return_t pgaspi_segment_max (gaspi_number_t * const segment_max);

  gaspi_return_t pgaspi_write (const gaspi_segment_id_t segment_id_local,
			       const gaspi_offset_t offset_local,
			       const gaspi_rank_t rank,
			       const gaspi_segment_id_t segment_id_remote,
			       const gaspi_offset_t offset_remote,
			       const gaspi_size_t size,
			       const gaspi_queue_id_t queue,
			       const gaspi_timeout_t timeout_ms);
  gaspi_return_t pgaspi_read (const gaspi_segment_id_t segment_id_local,
			      const gaspi_offset_t offset_local,
			      const gaspi_rank_t rank,
			      const gaspi_segment_id_t segment_id_remote,
			      const gaspi_offset_t offset_remote,
			      const gaspi_size_t size,
			      const gaspi_queue_id_t queue,
			      const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_write_list (const gaspi_number_t num,
				    gaspi_segment_id_t *
				    const segment_id_local,
				    gaspi_offset_t * const offset_local,
				    const gaspi_rank_t rank,
				    gaspi_segment_id_t *
				    const segment_id_remote,
				    gaspi_offset_t * const offset_remote,
				    gaspi_size_t * const size,
				    const gaspi_queue_id_t queue,
				    const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_read_list (const gaspi_number_t num,
				   gaspi_segment_id_t * const segment_id_local,
				   gaspi_offset_t * const offset_local,
				   const gaspi_rank_t rank,
				   gaspi_segment_id_t *
				   const segment_id_remote,
				   gaspi_offset_t * const offset_remote,
				   gaspi_size_t * const size,
				   const gaspi_queue_id_t queue,
				   const gaspi_timeout_t timeout_ms);
  gaspi_return_t pgaspi_wait (const gaspi_queue_id_t queue,
			      const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_barrier (const gaspi_group_t group,
				 const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_allreduce (const gaspi_pointer_t buffer_send,
				   gaspi_pointer_t const buffer_receive,
				   const gaspi_number_t num,
				   const gaspi_operation_t operation,
				   const gaspi_datatype_t datatyp,
				   const gaspi_group_t group,
				   const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_allreduce_user (const gaspi_pointer_t const buffer_send,
					gaspi_pointer_t const buffer_receive,
					const gaspi_number_t num,
					const gaspi_size_t element_size,
					gaspi_reduce_operation_t const
					reduce_operation,
					gaspi_reduce_state_t const reduce_state,
					const gaspi_group_t group,
					const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_atomic_fetch_add (const gaspi_segment_id_t segment_id,
					  const gaspi_offset_t offset,
					  const gaspi_rank_t rank,
					  const gaspi_atomic_value_t val_add,
					  gaspi_atomic_value_t * const val_old,
					  const gaspi_timeout_t timeout_ms);
  gaspi_return_t pgaspi_atomic_compare_swap (const gaspi_segment_id_t
					     segment_id,
					     const gaspi_offset_t offset,
					     const gaspi_rank_t rank,
					     const gaspi_atomic_value_t
					     comparator,
					     const gaspi_atomic_value_t
					     val_new,
					     gaspi_atomic_value_t *
					     const val_old,
					     const gaspi_timeout_t timeout_ms);
  gaspi_return_t pgaspi_passive_send (const gaspi_segment_id_t
				      segment_id_local,
				      const gaspi_offset_t offset_local,
				      const gaspi_rank_t rank,
				      const gaspi_size_t size,
				      const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_passive_receive (const gaspi_segment_id_t
					 segment_id_local,
					 const gaspi_offset_t offset_local,
					 gaspi_rank_t * const rem_rank,
					 const gaspi_size_t size,
					 const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_notify (const gaspi_segment_id_t segment_id_remote,
				const gaspi_rank_t rank,
				const gaspi_notification_id_t notification_id,
				const gaspi_notification_t notification_value,
				const gaspi_queue_id_t queue,
				const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_notify_waitsome (const gaspi_segment_id_t
					 segment_id_local,
					 const gaspi_notification_id_t
					 notification_begin,
					 const gaspi_number_t num,
					 gaspi_notification_id_t *
					 const first_id,
					 const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_notify_reset (const gaspi_segment_id_t
				      segment_id_local,
				      const gaspi_notification_id_t
				      notification_id,
				      gaspi_notification_t *
				      const old_notification_val);

  gaspi_return_t pgaspi_write_notify (const gaspi_segment_id_t
				      segment_id_local,
				      const gaspi_offset_t offset_local,
				      const gaspi_rank_t rank,
				      const gaspi_segment_id_t
				      segment_id_remote,
				      const gaspi_offset_t offset_remote,
				      const gaspi_size_t size,
				      const gaspi_notification_id_t
				      notification_id,
				      const gaspi_notification_t
				      notification_value,
				      const gaspi_queue_id_t queue,
				      const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_write_list_notify (const gaspi_number_t num,
					   gaspi_segment_id_t *
					   const segment_id_local,
					   gaspi_offset_t * const offset_local,
					   const gaspi_rank_t rank,
					   gaspi_segment_id_t *
					   const segment_id_remote,
					   gaspi_offset_t *
					   const offset_remote,
					   gaspi_size_t * const size,
					   const gaspi_segment_id_t
					   segment_id_notification,
					   const gaspi_notification_id_t
					   notification_id,
					   const gaspi_notification_t
					   notification_value,
					   const gaspi_queue_id_t queue,
					   const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_read_notify (const gaspi_segment_id_t segment_id_local,
				     const gaspi_offset_t offset_local,
				     const gaspi_rank_t rank,
				     const gaspi_segment_id_t segment_id_remote,
				     const gaspi_offset_t offset_remote,
				     const gaspi_size_t size,
				     const gaspi_notification_id_t notification_id,
				     const gaspi_queue_id_t queue,
				     const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_read_list_notify (const gaspi_number_t num,
					  gaspi_segment_id_t * const segment_id_local,
					  gaspi_offset_t * const offset_local,
					  const gaspi_rank_t rank,
					  gaspi_segment_id_t * const segment_id_remote,
					  gaspi_offset_t * const offset_remote,
					  gaspi_size_t * const size,
					  const gaspi_segment_id_t segment_id_notification,
					  const gaspi_notification_id_t notification_id,
					  const gaspi_queue_id_t queue,
					  const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_queue_size (const gaspi_queue_id_t queue,
				    gaspi_number_t * const queue_size);

  gaspi_return_t pgaspi_queue_num (gaspi_number_t * const queue_num);

  gaspi_return_t pgaspi_queue_create(gaspi_queue_id_t * const queue_id,
				     const gaspi_timeout_t timeout_ms);

  gaspi_return_t pgaspi_queue_delete(const gaspi_queue_id_t queue_id);

  gaspi_return_t pgaspi_queue_size_max (gaspi_number_t * const queue_size_max);

  gaspi_return_t pgaspi_transfer_size_min (gaspi_size_t *
					   const transfer_size_min);

  gaspi_return_t pgaspi_transfer_size_max (gaspi_size_t *
					   const transfer_size_max);

  gaspi_return_t pgaspi_notification_num (gaspi_number_t * const notification_num);

  gaspi_return_t pgaspi_passive_transfer_size_max (gaspi_size_t * const passive_transfer_size_max);

  gaspi_return_t pgaspi_allreduce_buf_size (gaspi_size_t * const buf_size);

  gaspi_return_t pgaspi_allreduce_elem_max (gaspi_number_t * const elem_max);

  gaspi_return_t pgaspi_rw_list_elem_max (gaspi_number_t * const elem_max);

  gaspi_return_t pgaspi_queue_max(gaspi_number_t * const queue_max);

  gaspi_return_t pgaspi_network_type (gaspi_network_t * const network_type);

  gaspi_return_t pgaspi_time_ticks (gaspi_cycles_t * const ticks);

  gaspi_return_t pgaspi_time_get (gaspi_time_t * const wtime);

  gaspi_return_t pgaspi_cpu_frequency (gaspi_float * const cpu_mhz);

  gaspi_return_t pgaspi_machine_type (char const machine_type[16]);

  gaspi_return_t pgaspi_state_vec_get (gaspi_state_vector_t state_vector);

  void pgaspi_printf (const char *fmt, ...);

  void pgaspi_print_affinity_mask (void);

  gaspi_return_t pgaspi_numa_socket(gaspi_uchar * const socket);


  gaspi_return_t pgaspi_set_socket_affinity (const gaspi_uchar socket);


  gaspi_return_t pgaspi_statistic_verbosity_level(gaspi_number_t _verbosity_level);

  gaspi_return_t pgaspi_statistic_counter_max(gaspi_statistic_counter_t* counter_max);


  gaspi_return_t
  pgaspi_statistic_counter_info(gaspi_statistic_counter_t counter
				, gaspi_statistic_argument_t* counter_argument
				, gaspi_string_t* counter_name
				, gaspi_string_t* counter_description
				, gaspi_number_t* verbosity_level
				);

  gaspi_return_t
  pgaspi_statistic_counter_get ( gaspi_statistic_counter_t counter
				 , gaspi_number_t argument
				 , unsigned long *value
				 );

  gaspi_return_t pgaspi_statistic_counter_reset (gaspi_statistic_counter_t counter);

  gaspi_string_t pgaspi_error_str(gaspi_return_t error_code);

#ifdef __cplusplus
}
#endif

#endif //PGPI2_H
