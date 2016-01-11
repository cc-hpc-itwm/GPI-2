/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2015

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

/********************************************************************************************
 *
 *
 *                                  GASPI Extensions
 *
 *          Functionality that is not (yet) present in the GASPI specification.
 *
 *
 ********************************************************************************************/

#ifdef __cplusplus
extern "C"
{
#endif

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

  /** Create a new communication queue.
   *
   *
   *
   * @param queue Output parameter with id of created queue.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_queue_create(gaspi_queue_id_t *queue, gaspi_timeout_t timeout_ms);

  /** Delete a new communication queue.
   *
   *
   *
   * @param queue The queue ID to delete.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_queue_delete(gaspi_queue_id_t queue);

#ifdef __cplusplus
}
#endif

#endif //GPI2_EXT_H  
