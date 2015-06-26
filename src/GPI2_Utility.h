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

#ifndef GPI2_UTILITY_H
#define GPI2_UTILITY_H 1

#include <errno.h>
#include <sched.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <xmmintrin.h>

#ifdef MIC
#define gaspi_delay() _mm_delay_32(32)
#else
#define gaspi_delay() _mm_pause()
#endif


#define MAX(a,b)  (((a)<(b)) ? (b) : (a))
#define MIN(a,b)  (((a)>(b)) ? (b) : (a))



#ifdef DEBUG
#include "GPI2.h"
#define gaspi_print_error(msg, ...)					\
  int errsv = errno;							\
  if( errsv != 0 )							\
    fprintf(stderr,"[Rank %4u]: Error %d (%s) at (%s:%d):" msg "\n",	\
	    glb_gaspi_ctx.rank, errsv, (char *) strerror(errsv),	\
	    __FILE__, __LINE__, ##__VA_ARGS__);				\
  else									\
    fprintf(stderr,"[Rank %4u]: Error at (%s:%d):" msg "\n",		\
	    glb_gaspi_ctx.rank, __FILE__, __LINE__, ##__VA_ARGS__)


#define gaspi_print_warning(msg, ...)					\
  fprintf(stderr,"[Rank %4u]: Warning:" msg "\n", glb_gaspi_ctx.rank, ##__VA_ARGS__)

#define gaspi_verify_null_ptr(ptr)				\
  {								\
  if(ptr == NULL)						\
    {								\
      gaspi_print_error ("Passed argument is a NULL pointer");	\
      return GASPI_ERR_NULLPTR;					\
    }								\
  }

#define gaspi_verify_rank(rank)						\
  {									\
    if(rank >= glb_gaspi_ctx.tnc)					\
      return GASPI_ERR_INV_RANK;					\
  }

#define gaspi_verify_queue(queue)					\
  {									\
    if(queue > glb_gaspi_ctx.num_queues - 1)				\
      return GASPI_ERR_INV_QUEUE;					\
  }

#define gaspi_verify_queue_depth(depth)					\
  {									\
    if(depth >= glb_gaspi_cfg.queue_depth)				\
      return GASPI_ERR_MANY_Q_REQS;					\
  }

#define gaspi_verify_segment(seg_id)					\
  {									\
    if( seg_id >= GASPI_MAX_MSEGS)					\
      return GASPI_ERR_INV_SEG;						\
  }

#define gaspi_verify_unaligned_off(offset)				\
  {									\
    if( offset & 0x7 )							\
      return GASPI_ERR_UNALIGN_OFF;					\
  }

#define gaspi_verify_local_off(off, seg_id)				\
  {									\
  gaspi_verify_segment(seg_id);						\
  gaspi_verify_null_ptr(glb_gaspi_ctx.rrmd[seg_id]);			\
  if( off > glb_gaspi_ctx.rrmd[seg_id][glb_gaspi_ctx.rank].size)	\
    return GASPI_ERR_INV_LOC_OFF;					\
  }

#define gaspi_verify_remote_off(off, seg_id, rank)			\
  {									\
  gaspi_verify_segment(seg_id);						\
  gaspi_verify_null_ptr(glb_gaspi_ctx.rrmd[seg_id]);			\
  gaspi_verify_rank(rank);						\
  if( off > glb_gaspi_ctx.rrmd[seg_id][rank].size)			\
    return GASPI_ERR_INV_REM_OFF;					\
  }

#define gaspi_verify_comm_size(sz, seg_id_loc, seg_id_rem, rnk, max)	\
  {									\
    if( sz < 1								\
	|| sz > max							\
	|| sz > glb_gaspi_ctx.rrmd[seg_id_rem][rnk].size		\
	|| sz > glb_gaspi_ctx.rrmd[seg_id_loc][glb_gaspi_ctx.rank].size) \
      {									\
	return GASPI_ERR_INV_COMMSIZE;					\
      }									\
  }

#define gaspi_verify_segment_size(size)			\
  {							\
    if(0 == size)					\
      {							\
	return GASPI_ERR_INV_SEGSIZE;			\
      }							\
  }

#define gaspi_verify_group(group)					\
  {									\
    if(group >= GASPI_MAX_GROUPS || glb_gaspi_group_ctx[group].id < 0)	\
      return GASPI_ERR_INV_GROUP;					\
  }

#define gaspi_verify_init(funcname)					\
  {									\
    if(!glb_gaspi_init)							\
      {									\
	gaspi_print_error("Error: Invalid function (%s) before initialization", \
			  funcname);					\
	  return GASPI_ERR_NOINIT;					\
      }									\
  }


#else
#define gaspi_print_error(msg, ...)
#define gaspi_print_warning(msg, ...)
#define gaspi_verify_null_ptr(ptr)
#define gaspi_verify_rank(rank)
#define gaspi_verify_group(grp)
#define gaspi_verify_queue(queue)
#define gaspi_verify_queue_depth(depth)
#define gaspi_verify_segment(seg_id)
#define gaspi_verify_unaligned_off(offset)
#define gaspi_verify_local_off(off, seg_id)
#define gaspi_verify_remote_off(off, seg_id, rank)
#define gaspi_verify_comm_size(size, seg_id_loc, seg_id_rem, rank, max)
#define gaspi_verify_segment_size(size)
#define gaspi_verify_init(funcname)
#endif

#define gaspi_verify_setup(funcname)					\
  {									\
    if(glb_gaspi_init)							\
      {									\
	gaspi_print_error("Error: Invalid function (%s) after initialization", \
		funcname);						\
	return GASPI_ERR_INITED;					\
      }									\
  }

ulong gaspi_load_ulong(volatile ulong *ptr);
float gaspi_get_cpufreq ();
int gaspi_get_affinity_mask (const int sock, cpu_set_t * cpuset);

char * gaspi_get_hn (const unsigned int id);

static inline int gaspi_thread_sleep(int msecs)
{
  struct timespec sleep_time, rem;
  sleep_time.tv_sec = msecs / 1000;
  sleep_time.tv_nsec = 0;// msecs * 1000000;

  return nanosleep(&sleep_time, &rem);
}

#endif
