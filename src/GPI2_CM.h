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

#ifndef _GPI2_CM_H_
#define _GPI2_CM_H_

#include "PGASPI.h"


typedef enum
  {
    GASPI_ENDPOINT_DISCONNECTED = 0,
    GASPI_ENDPOINT_CONNECTED = 1
  }gaspi_endpoint_conn_status_t;

typedef enun
  {
    GASPI_ENDPOINT_NOT_CREATED = 0,
    GASPI_ENDPOINT_CREATED = 1
  }gaspi_endpoint_creation_status_t;

typedef enum
  {
    GASPI_QUEUE_STATE_DISABLED = 0,
    GASPI_QUEUE_STATE_CREATED = 1,
    GASPI_QUEUE_STATE_ENABLED = 2,
  } gaspi_queue_connection_state_t ;

/* connection to a endpoint */
typedef struct
{
  gaspi_rank_t rank; /* to whom */
  volatile gaspi_endpoint_creation_status_t istat;
  volatile gaspi_endpoint_conn_status_t cstat;
  gaspi_queue_connection_state_t queue_state[GASPI_MAX_QP];
} gaspi_endpoint_conn_t;

gaspi_return_t
pgaspi_create_endpoint_to(const gaspi_rank_t, const gaspi_timeout_t);

gaspi_return_t
pgaspi_connect_endpoint_to(const gaspi_rank_t, const gaspi_timeout_t);

gaspi_return_t
pgaspi_local_disconnect(const gaspi_rank_t, const gaspi_timeout_t);


#endif /* _GPI2_CM_H_ */
