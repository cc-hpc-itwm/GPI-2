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

#ifndef _GPI2_THREADS_H_
#define _GPI2_THREADS_H_

#include "GASPI.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /** Get thread identifier
   *
   * 
   * @param Output parameter with thread identifier
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_get_tid(gaspi_int * const tid);

  /** Get total number of threads
   * 
   * 
   * @param Output parameter with total number of threads
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_get_total(gaspi_int *const num);
  
  
  /** Get total number of available cpu cores
   * 
   * 
   * @param cores Output paramter with the number of cores.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_get_num_cores(gaspi_int * const cores);

  /** Initialize threads (in all available cores)
   * 
   * 
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_init(gaspi_int * const num);

  //returns activated cores (specified by caller)

  /** Initialize threads (a particular number of threads)
   * 
   * 
   * @param use_nr_of_threads Number of threads to start.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_init_user(const unsigned int use_nr_of_threads);

  /** Finalize threads
   * 
   * 
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_term(void);

  /** Run a particular task (function)
   * 
   * 
   * @param function The function to run.
   * @param arg The arguments of the function to run.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_run(void* (*function)(void*), void *arg);

  /** Register a thread with the pool.
   * 
   * 
   * @param tid Output parameter with the thread identifier.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_register(gaspi_int * tid);

  /** Synchronize all local threads (local barrier).
   *
   * 
   */
  void gaspi_threads_sync(void);

  /** Synchronize all threads in a group (global barrier).
   * Implies a gaspi_barrier within the group.
   *
   * @param group The group involved in the barrier. 
   * @param timeout The timeout to be applied in the global barrier(gaspi_barrier).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_threads_sync_all(const gaspi_group_t g, const gaspi_timeout_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif
