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

#ifndef GPI2_GRP_H_
#define GPI2_GRP_H_

#include "GPI2_Types.h"

typedef enum{
  GASPI_BARRIER = 1,
  GASPI_ALLREDUCE = 2,
  GASPI_ALLREDUCE_USER = 4,
  GASPI_NONE = 7
}gaspi_async_coll_t;

typedef struct
{
  int id;
  gaspi_lock_t gl;
  gaspi_lock_t del;
  volatile unsigned char barrier_cnt;
  volatile unsigned char togle;
  gaspi_async_coll_t coll_op;
  int lastmask;
  int level, tmprank, dsize, bid;
  int rank, tnc;
  int next_pof2;
  int pof2_exp;
  int *rank_grp;
  int *committed_rank;
  gaspi_rc_mseg *rrcd;
} gaspi_group_ctx;

gaspi_return_t
pgaspi_group_all_local_create(const gaspi_timeout_t timeout_ms);

#endif /* GPI2_GRP_H_ */
