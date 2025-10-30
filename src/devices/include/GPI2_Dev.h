/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2024

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
#include "GPI2_Types.h"

/* Device interface */
int pgaspi_dev_init_core (gaspi_context_t * const);

int pgaspi_dev_cleanup_core (gaspi_context_t * const);

int
pgaspi_dev_register_mem (gaspi_context_t const *const,
                         gaspi_rc_mseg_t * seg);

int
pgaspi_dev_unregister_mem (gaspi_context_t const *const,
                           gaspi_rc_mseg_t *);

int pgaspi_dev_connect_context (gaspi_context_t const *const,
                                const int);

int pgaspi_dev_disconnect_context (gaspi_context_t * const,
                                   const int);

int
pgaspi_dev_create_endpoint (gaspi_context_t const *const,
                            const int,
                            void **,
                            void **,
                            size_t *);

int
pgaspi_dev_comm_queue_delete (gaspi_context_t const *const,
                              const unsigned int);

int
pgaspi_dev_comm_queue_create (gaspi_context_t const *const,
                              const unsigned int,
                              const unsigned short);

int
pgaspi_dev_comm_queue_is_valid (gaspi_context_t const *const gctx,
                                const unsigned int id);

int
pgaspi_dev_comm_queue_connect (gaspi_context_t const *const,
                               const unsigned short,
                               const int i);

/* Device interface (GASPI routines) */

/* Groups */
int pgaspi_dev_poll_groups (gaspi_context_t * const);

int
pgaspi_dev_post_group_write (gaspi_context_t * const,
                             void *,
                             int,
                             int,
                             void *,
                             unsigned char);

int pgaspi_dev_queue_size (const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_purge (gaspi_context_t * const,
                  const gaspi_queue_id_t,
                  const gaspi_timeout_t);

gaspi_return_t
pgaspi_dev_write (gaspi_context_t * const,
                  const gaspi_segment_id_t,
                  const gaspi_offset_t,
                  const gaspi_rank_t,
                  const gaspi_segment_id_t,
                  const gaspi_offset_t,
                  const gaspi_size_t,
                  const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read (gaspi_context_t * const,
                 const gaspi_segment_id_t,
                 const gaspi_offset_t,
                 const gaspi_rank_t,
                 const gaspi_segment_id_t,
                 const gaspi_offset_t,
                 const gaspi_size_t,
                 const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_wait (gaspi_context_t * const,
                 const gaspi_queue_id_t,
                 const gaspi_timeout_t);

gaspi_return_t
pgaspi_dev_write_list (gaspi_context_t * const,
                       const gaspi_number_t,
                       gaspi_segment_id_t * const,
                       gaspi_offset_t * const,
                       const gaspi_rank_t,
                       gaspi_segment_id_t * const,
                       gaspi_offset_t * const,
                       gaspi_size_t * const,
                       const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read_list (gaspi_context_t * const,
                      const gaspi_number_t,
                      gaspi_segment_id_t * const,
                      gaspi_offset_t * const,
                      const gaspi_rank_t,
                      gaspi_segment_id_t * const,
                      gaspi_offset_t * const,
                      gaspi_size_t * const,
                      const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_notify (gaspi_context_t * const,
                   const gaspi_segment_id_t,
                   const gaspi_rank_t,
                   const gaspi_notification_id_t,
                   const gaspi_notification_t,
                   const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_write_notify (gaspi_context_t * const,
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
pgaspi_dev_write_list_notify (gaspi_context_t * const,
                              const gaspi_number_t,
                              gaspi_segment_id_t * const,
                              gaspi_offset_t * const,
                              const gaspi_rank_t,
                              gaspi_segment_id_t * const,
                              gaspi_offset_t * const,
                              gaspi_size_t * const,
                              const gaspi_segment_id_t,
                              const gaspi_notification_id_t,
                              const gaspi_notification_t,
                              const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read_notify (gaspi_context_t * const,
                        const gaspi_segment_id_t,
                        const gaspi_offset_t,
                        const gaspi_rank_t,
                        const gaspi_segment_id_t,
                        const gaspi_offset_t,
                        const gaspi_size_t,
                        const gaspi_notification_id_t,
                        const gaspi_queue_id_t);

gaspi_return_t
pgaspi_dev_read_list_notify (gaspi_context_t * const,
                             const gaspi_number_t,
                             gaspi_segment_id_t * const,
                             gaspi_offset_t * const,
                             const gaspi_rank_t,
                             gaspi_segment_id_t * const,
                             gaspi_offset_t * const,
                             gaspi_size_t * const,
                             const gaspi_segment_id_t,
                             const gaspi_notification_id_t,
                             const gaspi_queue_id_t);
gaspi_return_t
pgaspi_dev_atomic_fetch_add (gaspi_context_t * const,
                             const gaspi_segment_id_t,
                             const gaspi_offset_t,
                             const gaspi_rank_t,
                             const gaspi_atomic_value_t);

gaspi_return_t
pgaspi_dev_atomic_compare_swap (gaspi_context_t * const,
                                const gaspi_segment_id_t,
                                const gaspi_offset_t,
                                const gaspi_rank_t,
                                const gaspi_atomic_value_t,
                                const gaspi_atomic_value_t);

gaspi_return_t
pgaspi_dev_passive_send (gaspi_context_t * const,
                         const gaspi_segment_id_t,
                         const gaspi_offset_t,
                         const gaspi_rank_t,
                         const gaspi_size_t,
                         const gaspi_timeout_t);

gaspi_return_t
pgaspi_dev_passive_receive (gaspi_context_t * const,
                            const gaspi_segment_id_t,
                            const gaspi_offset_t,
                            gaspi_rank_t * const,
                            const gaspi_size_t,
                            const gaspi_timeout_t);

#ifdef GPI2_DEVICE_OFI
uint64_t
pgaspi_dev_get_mr_rkey (gaspi_context_t const *const gctx,
                        void* mr,
                        gaspi_rank_t rank);
#endif

#endif //_GPI2_DEV_H_
