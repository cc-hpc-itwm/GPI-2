/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2023

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

#ifndef _GPI2_TCP_H_
#define _GPI2_TCP_H_

#include "GPI2.h"
#include "GPI2_Dev.h"

#include "tcp_device.h"

#define QP_MAX_NUM 4096

typedef struct
{

  /* Groups communication */
  struct tcp_cq *scqGroups, *rcqGroups;
  struct tcp_queue *qpGroups;

  /* Passive communication */
  int srqP;                   /* passive comm (local conn) */
  struct tcp_passive_channel *channelP;
  struct tcp_queue *qpP;
  struct tcp_cq *scqP;
  struct tcp_cq *rcqP;

  /* Queues communication */
  struct tcp_cq *scqC[GASPI_MAX_QP];
  struct tcp_queue *qpC[GASPI_MAX_QP];

  int device_channel;

} gaspi_tcp_ctx;

#endif //_GPI2_TCP_H_
