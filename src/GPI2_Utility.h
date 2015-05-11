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

#ifdef DEBUG
#include "GPI2.h"
#define gaspi_print_error(msg, ...)					\
  int errsv = errno;							\
  if( errsv != 0 )							\
    fprintf(stderr,"[Rank %4u]: Error %d (%s) at (%s:%d):" msg "\n",glb_gaspi_ctx.rank, errsv, (char *) strerror(errsv), __FILE__, __LINE__, ##__VA_ARGS__); \
  else									\
    fprintf(stderr,"[Rank %4u]: Error at (%s:%d):" msg "\n",glb_gaspi_ctx.rank, __FILE__, __LINE__, ##__VA_ARGS__)

#define gaspi_print_warning(msg, ...)					\
  fprintf(stderr,"[Rank %4u]: Warning:" msg "\n",glb_gaspi_ctx.rank, ##__VA_ARGS__)

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

#define gaspi_verify_segment(seg_id)					\
  {									\
    if( seg_id >= GASPI_MAX_MSEGS)					\
      return GASPI_ERR_INV_SEG;						\
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


#else
#define gaspi_print_error(msg, ...)
#define gaspi_print_warning(msg, ...)
#define gaspi_verify_null_ptr(ptr)
#define gaspi_verify_rank(rank)
#define gaspi_verify_segment(seg_id)
#define gaspi_verify_segment_size(size)
#define gaspi_verify_group(group)
#endif

#ifdef MIC
#define gaspi_delay() _mm_delay_32(32)
#else
#define gaspi_delay() _mm_pause()
#endif


#define MAX(a,b)  (((a)<(b)) ? (b) : (a))
#define MIN(a,b)  (((a)>(b)) ? (b) : (a))

#define gaspi_verify_init(funcname)					\
  {									\
    if(!glb_gaspi_init)							\
      {									\
	gaspi_print_error("Error: Invalid function (%s) before initialization", \
			  funcname);					\
	  return GASPI_ERR_NOINIT;					\
      }									\
  }

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
