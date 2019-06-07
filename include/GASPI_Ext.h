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

#ifndef GPI2_EXT_H
#define GPI2_EXT_H

#include "GASPI.h"

/**********************************************************************************************
 *
 *
 *                                    GASPI Extensions
 *
 *     Functionality that is not present in the GASPI specification (subject to be changed)
 *
 *
 **********************************************************************************************/

#ifdef __cplusplus
extern "C"
{
#endif

  /** Check if GPI-2 is initialized
   *
   * @param initialized Output parameter with flag value.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error.
   */
  gaspi_return_t gaspi_initialized (gaspi_number_t * initialized);

  /** Get the process local rank.
   *
   *
   * @param local_rank Rank within a node of calling process.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_proc_local_rank (gaspi_rank_t * const local_rank);

  /** Get the number of processes (ranks) started by the application.
   *
   *
   * @param local_num The number of processes (ranks) in the same node
   *
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_proc_local_num (gaspi_rank_t * const local_num);

  /** Get the machine type (CPU, accelerator...)
   *
   *
   * @param machine_type Output parameter with machine type.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_machine_type (char const machine_type[16]);

    /** Get the CPU frequency.
   *
   *
   * @param cpu_mhz Output parameter with the frequency.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_cpu_frequency (gaspi_float * const cpu_mhz);

  /** GASPI printf to print the gaspi_logger.
   *
   *
   * @param fmt printf parameters.
   */
  void gaspi_printf (const char *fmt, ...);

  /** GASPI printf to print to a particular gaspi_logger ie. a
   * gaspi_logger running on the node of a particular rank.
   *
   * @param rank the rank of the logger node.
   * @param fmt printf parameters.
   */
  void gaspi_printf_to (gaspi_rank_t rank, const char *fmt, ...);

  /**  Print the CPU's affinity mask.
   *
   *
   */
  void gaspi_print_affinity_mask (void);

  /** Get NUMA socket
   *
   *
   * @param socket Output parameter with the socket
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case
   * GPI2 was not started with NUMA enabled.
   */
  gaspi_return_t gaspi_numa_socket(gaspi_uchar * const socket);

  /** Set socket affinity
   *
   *
   */
  gaspi_return_t gaspi_set_socket_affinity (const gaspi_uchar socket);

  /** Get string describing return value. This is slightly more
   * practical than gaspi_print_error.
   *
   *
   * @param error_code The return value to be described.
   *
   * @return A string that describes the return value.
   */
  gaspi_string_t gaspi_error_str(gaspi_return_t error_code);

  /** Ping a particular proc (rank).
   * This is useful in FT applications to determine if a rank is alive.
   *
   *
   * @param rank The rank to ping.
   * @param tout A timeout value in milliseconds.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_proc_ping (const gaspi_rank_t rank, gaspi_timeout_t tout);

  /** Get an available segment id (only locally).
   *
   * To create/alloc a segment, the application must provide a segment
   * id. This provides a helper function to find the next available id
   * locally i.e. for the calling rank.
   *
   *
   * @param avail_seg_id The available segment id.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_avail_local (gaspi_segment_id_t* const avail_seg_id);

  /** Get the size of a given segment on a particular rank.
   *
   *
   * @param segment_id The segment id we are interested in.
   * @param rank The rank.
   * @param size Output parameter with the size of the segment.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_size (const gaspi_segment_id_t segment_id,
				     const gaspi_rank_t rank,
				     gaspi_size_t * const size);

  /** Get the maximum number of elements allowed in list (read, write)
   * operations.
   *
   *
   * @param elem_max Output parameter with the maximum number of elements.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_rw_list_elem_max (gaspi_number_t * const elem_max);


#ifdef __cplusplus
}
#endif

#endif //GPI2_EXT_H
