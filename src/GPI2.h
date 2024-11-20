/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2024

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

#include "GASPI_types.h"
#include "GPI2_Sys.h"
#include "GPI2_Types.h"

extern gaspi_context_t glb_gaspi_ctx;

#define GASPI_ATOMIC_TRY_LOCK(l) __sync_lock_test_and_set (l, 1)
#define GASPI_ATOMIC_UNLOCK(l)  __sync_lock_release (l)

static inline void
lock_gaspi (gaspi_lock_t * l)
{
  while (GASPI_ATOMIC_TRY_LOCK (&l->lock))
  {
    while (l->lock)
    {
      GASPI_DELAY();
    }
  }
}

static inline int
lock_gaspi_tout (gaspi_lock_t * l, const gaspi_timeout_t timeout_ms)
{

  if (timeout_ms == GASPI_BLOCK)
  {
    while (GASPI_ATOMIC_TRY_LOCK (&l->lock))
    {
      while (l->lock)
      {
        GASPI_DELAY();
      }
    }
    return 0;
  }
  else if (timeout_ms == GASPI_TEST)
  {
    const unsigned char val = GASPI_ATOMIC_TRY_LOCK (&l->lock);

    return val;
  }

  //timeout
  const gaspi_cycles_t s0 = gaspi_get_cycles();

  while (GASPI_ATOMIC_TRY_LOCK (&l->lock))
  {
    while (l->lock)
    {
      const gaspi_cycles_t s1 = gaspi_get_cycles();
      const gaspi_cycles_t tdelta = s1 - s0;

      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;

      if (ms > (float) timeout_ms)
      {
        return 1;
      }

      GASPI_DELAY();
    }
  }

  return 0;
}

static inline void
unlock_gaspi (gaspi_lock_t * l)
{
  GASPI_ATOMIC_UNLOCK (&l->lock);
}

#endif //_GPI2_H_
