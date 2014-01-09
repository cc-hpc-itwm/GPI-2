/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013

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

#ifndef _GPI2_H_
#define _GPI2_H_

#include "GASPI.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

#define COLL_MEM_SEND     (131136)
#define COLL_MEM_RECV     (COLL_MEM_SEND + 73728)
#define NEXT_OFFSET       (COLL_MEM_RECV + 73728)
#define NOTIFY_OFFSET     (65536*4)

gaspi_context glb_gaspi_ctx;

volatile int glb_gaspi_init;
volatile int glb_gaspi_sn_init;
volatile int glb_gaspi_ib_init;
volatile int gaspi_master_topo_data;
volatile int gaspi_rrcd_data;

//locks
gaspi_lock_t glb_gaspi_ctx_lock;
gaspi_lock_t gaspi_create_lock;
gaspi_lock_t gaspi_ccontext_lock;
gaspi_lock_t gaspi_mseg_lock;

static inline gaspi_cycles_t
gaspi_get_cycles ()
{
  unsigned low, high;
  unsigned long long val;

  asm volatile ("rdtsc":"=a" (low), "=d" (high));
  val = high;
  val = (val << 32) | low;
  return val;
}

#ifdef MIC
static inline
unsigned char gaspi_atomic_xchg (volatile unsigned char *addr,
				      const char new_val)
{
  unsigned char res;
  asm volatile ("lock; xchgb %0, %1":"+m" (*addr),
		"=a" (res):"1" (new_val):"memory");
  return res;
}

static inline
int lock_gaspi_tout (gaspi_lock_t * l, const unsigned int timeout_ms)
{

  if (timeout_ms == GASPI_BLOCK)
    {
      while (gaspi_atomic_xchg (&l->lock, 1))
	while (l->lock)
	  gaspi_delay ();
      return 0;
    }
  else if (timeout_ms == GASPI_TEST)
    {
      const unsigned char val = gaspi_atomic_xchg (&l->lock, 1);
      return val;
    }

  //timeout
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  while (gaspi_atomic_xchg (&l->lock, 1))
    {
      while (l->lock)
	{
	  const gaspi_cycles_t s1 = gaspi_get_cycles ();
	  const gaspi_cycles_t tdelta = s1 - s0;

	  const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
	  if (ms > (float) timeout_ms)
	    {
	      return 1;
	    }

	  gaspi_delay ();
	}
    }

  return 0;
}

static inline
void unlock_gaspi (gaspi_lock_t * l)
{
  gaspi_atomic_xchg (&l->lock, 0);
}

#else //!MIC

static inline
int lock_gaspi_tout (gaspi_lock_t * l, const unsigned int timeout_ms)
{

  if (timeout_ms == GASPI_BLOCK)
    {
      while (__sync_lock_test_and_set (&l->lock, 1))
	while (l->lock)
	  gaspi_delay ();
      return 0;
    }
  else if (timeout_ms == GASPI_TEST)
    {
      const unsigned char val = __sync_lock_test_and_set (&l->lock, 1);
      return val;
    }

  //timeout
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  while (__sync_lock_test_and_set (&l->lock, 1))
    {
      while (l->lock)
	{
	  const gaspi_cycles_t s1 = gaspi_get_cycles ();
	  const gaspi_cycles_t tdelta = s1 - s0;

	  const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
	  if (ms > (float) timeout_ms)
	    {
	      return 1;
	    }

	  gaspi_delay ();
	}
    }

  return 0;
}

static inline
void unlock_gaspi (gaspi_lock_t * l)
{
  __sync_lock_release (&l->lock);
}

#endif // MIC

char * gaspi_get_hn (const unsigned int id);

#endif //_GPI2_H_
