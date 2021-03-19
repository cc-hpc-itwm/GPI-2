/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2021

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

#ifndef _GPI2_CM_H_
#define _GPI2_CM_H_

#include "PGASPI.h"

typedef enum
{
  GASPI_ENDPOINT_DISCONNECTED = 0,
  GASPI_ENDPOINT_CONNECTED = 1
} gaspi_endpoint_conn_status_t;

typedef enum
{
  GASPI_ENDPOINT_NOT_CREATED = 0,
  GASPI_ENDPOINT_CREATED = 1
} gaspi_endpoint_creation_status_t;

/* Devices may have to exchange information when setting up a
   connection. */
typedef struct
{
  void *local_info;
  void *remote_info;
  size_t info_size;
} gaspi_dev_exch_info_t;

/* connection to a endpoint */
typedef struct
{
  gaspi_rank_t rank; /* to whom */
  gaspi_dev_exch_info_t exch_info; /* device exchange info */

  volatile gaspi_endpoint_creation_status_t istat;
  volatile gaspi_endpoint_conn_status_t cstat;
} gaspi_endpoint_conn_t;

gaspi_return_t
pgaspi_create_endpoint_to (const gaspi_rank_t rank,
                           gaspi_dev_exch_info_t * info,
                           const gaspi_timeout_t timeout_ms);

gaspi_return_t
pgaspi_connect_endpoint_to (const gaspi_rank_t target,
                            const gaspi_timeout_t timeout_ms);

gaspi_return_t
pgaspi_local_disconnect (const gaspi_rank_t from,
                         const gaspi_timeout_t timeout_ms);

#endif /* _GPI2_CM_H_ */
