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

#ifndef GPI2_UTILITY_H
#define GPI2_UTILITY_H 1

#include <errno.h>
#include <sched.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#ifdef MIC
#define GASPI_DELAY() _mm_delay_32(32)
#elif defined(__x86_64__)
#define GASPI_DELAY() _mm_pause()
#elif defined(__aarch64__)
#define GASPI_DELAY()  __asm__ __volatile__("yield")
#elif defined (__PPC64__)
#define GASPI_DELAY() __asm__ volatile("ori 0,0,0" ::: "memory");
#endif

#define MAX(a,b)  (((a)<(b)) ? (b) : (a))
#define MIN(a,b)  (((a)>(b)) ? (b) : (a))

#ifdef __GNUC__
#  define GASPI_UNUSED(x) UNUSED_ ## x __attribute__((__unused__))
#else
#  define GASPI_UNUSED(x) UNUSED_ ## (void)(x)
#endif

#ifdef DEBUG
#include "GPI2.h"
extern gaspi_config_t glb_gaspi_cfg;
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
    fprintf(stderr,"[Rank %4u]: Warning: " msg "\n", glb_gaspi_ctx.rank, ##__VA_ARGS__);	\
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

#define GASPI_VERIFY_QUEUE(queue)               \
  {                                             \
    if (queue > glb_gaspi_ctx.num_queues - 1)   \
    {                                           \
      return GASPI_ERR_INV_QUEUE;               \
    }                                           \
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
    if (sz < min                                                        \
        || sz > max                                                     \
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

float gaspi_get_cpufreq (void);

ulong gaspi_load_ulong (volatile ulong * ptr);

int gaspi_get_affinity_mask (const int sock, cpu_set_t * cpuset);

char *pgaspi_gethostname (const unsigned int id);

static inline int
gaspi_thread_sleep (int msecs)
{
  struct timespec sleep_time, rem;

  sleep_time.tv_sec = msecs / 1000;
  sleep_time.tv_nsec = 0;       // msecs * 1000000;

  return nanosleep (&sleep_time, &rem);
}

#endif //GPI2_UTILITY_H
