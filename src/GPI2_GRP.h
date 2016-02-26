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

#ifndef GPI2_GRP_H_
#define GPI2_GRP_H_

#include "GPI2_Types.h"

#define GPI2_REDUX_BUF_SIZE 2048

typedef enum {
  GASPI_BARRIER = 1,
  GASPI_ALLREDUCE = 2,
  GASPI_ALLREDUCE_USER = 4,
  GASPI_NONE = 7
} gaspi_async_coll_t;

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

#define GASPI_RESET_GROUP(group_ctx, i)					\
  do {									\
    group_ctx[i].rrcd = NULL;						\
    group_ctx[i].rank_grp = NULL;					\
    group_ctx[i].committed_rank = NULL;					\
    group_ctx[i].id = -1;						\
    group_ctx[i].togle = 0;						\
    group_ctx[i].barrier_cnt = 0;					\
    group_ctx[i].rank = 0;						\
    group_ctx[i].tnc = 0;						\
    group_ctx[i].coll_op = GASPI_NONE;					\
    group_ctx[i].lastmask = 0x1;					\
    group_ctx[i].tmprank = 0;						\
    group_ctx[i].bid = 0;						\
    group_ctx[i].level = 0;						\
    group_ctx[i].dsize = 0;						\
    group_ctx[i].next_pof2 = 0;						\
    group_ctx[i].pof2_exp = 0;						\
  }  while(0);

gaspi_return_t
pgaspi_group_all_local_create(const gaspi_timeout_t timeout_ms);

#endif /* GPI2_GRP_H_ */
