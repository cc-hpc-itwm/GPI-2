/*
Copyright (c) Fraunhofer ITWM, 2013-2025

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

#ifndef GPI2_UTILITY_H
#define GPI2_UTILITY_H 1

#include <errno.h>
#include <stdio.h>
#include <string.h>

#define MAX(a,b)  (((a)<(b)) ? (b) : (a))
#define MIN(a,b)  (((a)>(b)) ? (b) : (a))

#ifdef __GNUC__
#  define GASPI_UNUSED(x) UNUSED_ ## x __attribute__((__unused__))
#else
#  define GASPI_UNUSED(x) UNUSED_ ## (void)(x)
#endif

#ifdef DEBUG

#include "GPI2_Types.h"
extern gaspi_context_t glb_gaspi_ctx;
#define GASPI_DEBUG_PRINT_ERROR(msg, ...)                               \
  {                                                                     \
    int gaspi_debug_errsv = errno;                                      \
    if (gaspi_debug_errsv != 0)                                         \
    {                                                                   \
      fprintf(stderr,"[Rank %4u]: Error %d (%s) at (%s:%d):" msg "\n",  \
              glb_gaspi_ctx.rank, gaspi_debug_errsv, (char *) strerror(gaspi_debug_errsv), \
              __FILE__, __LINE__, ##__VA_ARGS__);                       \
    }                                                                   \
    else                                                                \
    {                                                                   \
      fprintf(stderr,"[Rank %4u]: Error at (%s:%d):" msg "\n",          \
              glb_gaspi_ctx.rank, __FILE__, __LINE__, ##__VA_ARGS__);   \
    }                                                                   \
    fflush(stderr);                                                     \
  }

#define GASPI_PRINT_WARNING(msg, ...)                                   \
  {                                                                     \
    fprintf(stderr,"[Rank %4u]: Warning: " msg "\n", glb_gaspi_ctx.rank, ##__VA_ARGS__);        \
  }

#define GASPI_VERIFY_NULL_PTR(ptr)                                      \
  {                                                                     \
    if (ptr == NULL)                                                    \
    {                                                                   \
      GASPI_DEBUG_PRINT_ERROR ("Passed argument is a NULL pointer");    \
      return GASPI_ERR_NULLPTR;                                         \
    }                                                                   \
  }

#define GASPI_VERIFY_RANK(rank)                 \
  {                                             \
    if (rank >= glb_gaspi_ctx.tnc)              \
    {                                           \
      return GASPI_ERR_INV_RANK;                \
    }                                           \
  }

#define GASPI_VERIFY_QUEUE(queue)                                              \
  {                                                                            \
    if (queue > GASPI_MAX_QP ||                                                \
        pgaspi_dev_comm_queue_is_valid (&glb_gaspi_ctx, queue))                \
    {                                                                          \
      return GASPI_ERR_INV_QUEUE;                                              \
    }                                                                          \
  }

#define GASPI_VERIFY_SEGMENT(seg_id)                      \
  {                                                       \
    if (seg_id >= glb_gaspi_ctx.config->segment_max)      \
    {                                                     \
      return GASPI_ERR_INV_SEG;                           \
    }                                                     \
  }

#define GASPI_VERIFY_UNALIGNED_OFF(offset)      \
  {                                             \
    if (offset & 0x7)                           \
    {                                           \
      return GASPI_ERR_UNALIGN_OFF;             \
    }                                           \
  }

#define GASPI_VERIFY_LOCAL_OFF(off, seg_id, sz)                         \
  {                                                                     \
    GASPI_VERIFY_SEGMENT(seg_id);                                       \
    GASPI_VERIFY_NULL_PTR(glb_gaspi_ctx.rrmd[seg_id]);                  \
    if (off >= glb_gaspi_ctx.rrmd[seg_id][glb_gaspi_ctx.rank].size)    \
    {                                                                   \
      return GASPI_ERR_INV_LOC_OFF;                                     \
    }                                                                   \
    if (off + sz > glb_gaspi_ctx.rrmd[seg_id][glb_gaspi_ctx.rank].size) \
    {                                                                   \
      return GASPI_ERR_INV_COMMSIZE;                                    \
    }                                                                   \
  }

#define GASPI_VERIFY_REMOTE_OFF(off, seg_id, rank, sz)          \
  {                                                             \
    GASPI_VERIFY_SEGMENT(seg_id);                               \
    GASPI_VERIFY_NULL_PTR(glb_gaspi_ctx.rrmd[seg_id]);          \
    GASPI_VERIFY_RANK(rank);                                    \
    if (off >= glb_gaspi_ctx.rrmd[seg_id][rank].size)           \
      return GASPI_ERR_INV_REM_OFF;                             \
    if (off + sz > glb_gaspi_ctx.rrmd[seg_id][rank].size)       \
    {                                                           \
      return GASPI_ERR_INV_COMMSIZE;                            \
    }                                                           \
  }

#define GASPI_VERIFY_COMM_SIZE(sz, seg_id_loc, seg_id_rem, rnk, min, max) \
  {                                                                     \
    if (sz > max                                                     \
        || sz > glb_gaspi_ctx.rrmd[seg_id_rem][rnk].size                \
        || sz > glb_gaspi_ctx.rrmd[seg_id_loc][glb_gaspi_ctx.rank].size) \
    {                                                                   \
      return GASPI_ERR_INV_COMMSIZE;                                    \
    }                                                                   \
  }

#define GASPI_VERIFY_SEGMENT_SIZE(size)         \
  {                                             \
    if (0 == size)                              \
    {                                           \
      return GASPI_ERR_INV_SEGSIZE;             \
    }                                           \
  }

#define GASPI_VERIFY_GROUP(group)                                       \
  {                                                                     \
    if (group >= glb_gaspi_ctx.config->group_max                        \
        || glb_gaspi_ctx.groups[group].id < 0)                          \
    {                                                                   \
      return GASPI_ERR_INV_GROUP;                                       \
    }                                                                   \
  }

#define GASPI_VERIFY_INIT(funcname)                                     \
  {                                                                     \
    if (!glb_gaspi_ctx.init)                                            \
    {                                                                   \
      GASPI_DEBUG_PRINT_ERROR("Error: Invalid function (%s) before initialization", \
                              funcname);                                \
      return GASPI_ERR_NOINIT;                                          \
    }                                                                   \
  }

#define GASPI_VERIFY_NOTIFICATION_NUM(notif_id)                                    \
  {                                                                                \
    if (notif_id >= glb_gaspi_ctx.config->notification_num)                        \
    {                                                                              \
      return GASPI_ERR_INV_NOTIF_ID;                                               \
    }                                                                              \
  }

#else

#define GASPI_DEBUG_PRINT_ERROR(msg, ...)
#define GASPI_PRINT_WARNING(msg, ...)
#define GASPI_VERIFY_NULL_PTR(ptr)
#define GASPI_VERIFY_RANK(rank)
#define GASPI_VERIFY_GROUP(grp)
#define GASPI_VERIFY_QUEUE(queue)
#define GASPI_VERIFY_QUEUE_SIZE_MAX(depth)
#define GASPI_VERIFY_SEGMENT(seg_id)
#define GASPI_VERIFY_UNALIGNED_OFF(offset)
#define GASPI_VERIFY_LOCAL_OFF(off, seg_id, sz)
#define GASPI_VERIFY_REMOTE_OFF(off, seg_id, rank, sz)
#define GASPI_VERIFY_COMM_SIZE(size, seg_id_loc, seg_id_rem, rank, min, max)
#define GASPI_VERIFY_SEGMENT_SIZE(size)
#define GASPI_VERIFY_INIT(funcname)
#define GASPI_VERIFY_NOTIFICATION_NUM(notif_id)

#endif //DEBUG

#define GASPI_VERIFY_SETUP(funcname)                                    \
  {                                                                     \
    if (glb_gaspi_ctx.init)                                             \
    {                                                                   \
      GASPI_DEBUG_PRINT_ERROR("Error: Invalid function (%s) after initialization", \
                              funcname);                                \
      return GASPI_ERR_INITED;                                          \
    }                                                                   \
  }
#endif //GPI2_UTILITY_H
