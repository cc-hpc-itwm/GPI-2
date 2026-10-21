/*
Copyright (c) Fraunhofer ITWM, 2013-2026

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

#include <stddef.h>
#include "GASPI_types.h"
#include "GPI2_Types.h"

typedef struct
{
  gaspi_group_t group;
  int tnc, cs, ret;
} gaspi_group_exch_info_t;


#define GASPI_RESET_GROUP(group_ctx, i)                                 \
  do {                                                                  \
    group_ctx[i].rrcd = NULL;                                           \
    group_ctx[i].rank_grp = NULL;                                       \
    group_ctx[i].committed_rank = NULL;                                 \
    group_ctx[i].id = -1;                                               \
    group_ctx[i].toggle = 0;                                            \
    group_ctx[i].barrier_cnt = 0;                                       \
    group_ctx[i].rank = 0;                                              \
    group_ctx[i].tnc = 0;                                               \
    group_ctx[i].active_coll_op = GASPI_NONE;                           \
    group_ctx[i].lastmask = 0x1;                                        \
    group_ctx[i].tmprank = 0;                                           \
    group_ctx[i].bid = 0;                                               \
    group_ctx[i].level = 0;                                             \
    group_ctx[i].dsize = 0;                                             \
    group_ctx[i].next_pof2 = 0;                                         \
    group_ctx[i].pof2_exp = 0;                                          \
  }  while(0);

gaspi_return_t
pgaspi_group_all_local_create (gaspi_context_t * const gctx,
                               const gaspi_timeout_t timeout_ms);

gaspi_return_t
pgaspi_group_delete_no_verify (const gaspi_group_t);

gaspi_group_exch_info_t*
pgaspi_group_create_exch_info (gaspi_group_t group,
             int tnc);

/* Sync flag of a collective step as 32-bit immediate data: value (bits
   0-7), toggle (bit 8), rank in group of the writer (bits 9-24) and
   group (bits 25-31). */
#define GPI2_GRP_SYNC_IMM_MAX_GROUPS (128)

static inline uint32_t
pgaspi_group_sync_imm (const gaspi_group_t group,
                       const int rank_in_group,
                       const unsigned char toggle,
                       const unsigned char value)
{
  return ((uint32_t) group << 25)
    | ((uint32_t) (rank_in_group & 0xffff) << 9)
    | ((uint32_t) (toggle & 0x1) << 8)
    | value;
}

/* Set the sync flag described by imm in the local group buffer */
void
pgaspi_group_sync_flag_set (gaspi_context_t * const gctx, const uint32_t imm);

#endif /* GPI2_GRP_H_ */
