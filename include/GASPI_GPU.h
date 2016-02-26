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

#ifndef GASPI_GPU_H
#define GASPI_GPU_H

#include <GASPI.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* Types */
  typedef int gaspi_gpu_t;
  typedef int gaspi_gpu_num;

  /** Initialization of GPUs.
   * It is a local synchronous blocking procedure.
   *
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_init_GPUs();

  /** Get GPU ids.
   * It is a local synchronous blocking procedure.
   *
   *
   * @param gpu_ids The address where to place the ids.
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */

  gaspi_return_t gaspi_GPU_ids(gaspi_gpu_t *gpu_ids);


  /** Get the number of available GPUs.
   * It is a local synchronous blocking procedure.
   *
   *
   * @param gpu The address where to place the number of gpus.
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */

  gaspi_return_t gaspi_number_of_GPUs(gaspi_gpu_num *gpus);

  /** One-sided write.
   *
   *
   * @param segment_id_local The local segment id with the data to write.
   * @param offset_local The local offset with the data to write.
   * @param rank The rank to which we want to write.
   * @param segment_id_remote The remote segment id to write to.
   * @param offset_remote The remote offset where to write to.
   * @param size The size of data to write.
   * @param queue The queue where to post the write request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_gpu_write(const gaspi_segment_id_t segment_id_local,
				 const gaspi_offset_t offset_local,
				 const gaspi_rank_t rank,
				 const gaspi_segment_id_t segment_id_remote,
				 const gaspi_offset_t offset_remote,
				 const gaspi_size_t size,
				 const gaspi_queue_id_t queue,
				 const gaspi_timeout_t timeout);

  /** Write data to a given node and notify it.
   *
   *
   * @param segment_id_local The segment identifier where data to be written is located.
   * @param offset_local The offset where the data to be written is located.
   * @param rank The rank where to write and notify.
   * @param segment_id_remote The remote segment identifier where to write the data to.
   * @param offset_remote The remote offset where to write to.
   * @param size The size of the data to write.
   * @param notification_id The notification identifier to use.
   * @param notification_value The notification value used.
   * @param queue The queue where to post the request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_gpu_write_notify(const gaspi_segment_id_t segment_id_local,
					const gaspi_offset_t offset_local,
					const gaspi_rank_t rank,
					const gaspi_segment_id_t segment_id_remote,
					const gaspi_offset_t offset_remote,
					const gaspi_size_t size,
					const gaspi_notification_id_t notification_id,
					const gaspi_notification_t notification_value,
					const gaspi_queue_id_t queue,
					const gaspi_timeout_t timeout);

#ifdef __cplusplus
}
#endif

#endif // GASPI_GPU_H
