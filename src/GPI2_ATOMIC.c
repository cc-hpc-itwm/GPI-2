/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2019

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

#include "PGASPI.h"
#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_Utility.h"

#ifdef GPI2_EXP_VERBS
#include <stdint.h>

uint64_t
swap_uint64( uint64_t val )
{
  val = ((val << 8) & 0xFF00FF00FF00FF00ULL ) | ((val >> 8) & 0x00FF00FF00FF00FFULL );
  val = ((val << 16) & 0xFFFF0000FFFF0000ULL ) | ((val >> 16) & 0x0000FFFF0000FFFFULL );

  return (val << 32) | (val >> 32);
}
#endif

#pragma weak gaspi_atomic_max = pgaspi_atomic_max
gaspi_return_t
pgaspi_atomic_max(gaspi_atomic_value_t *max_value)
{
  gaspi_verify_init("gaspi_atomic_max");
  gaspi_verify_null_ptr(max_value);

  *max_value = 0xffffffffffffffff;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_atomic_fetch_add = pgaspi_atomic_fetch_add
gaspi_return_t
pgaspi_atomic_fetch_add (const gaspi_segment_id_t segment_id,
			 const gaspi_offset_t offset,
			 const gaspi_rank_t rank,
			 const gaspi_atomic_value_t val_add,
			 gaspi_atomic_value_t * const val_old,
			 const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_atomic_fetch_add");
  gaspi_verify_remote_off(offset, segment_id, rank, sizeof(gaspi_atomic_value_t));
  gaspi_verify_null_ptr(val_old);
  gaspi_verify_unaligned_off(offset);

  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_return_t eret = GASPI_ERROR;

  if(lock_gaspi_tout (&gctx->groups[0].gl, timeout_ms))
    return GASPI_TIMEOUT;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if ( eret != GASPI_SUCCESS)
	{
	  goto endL;
	}
    }

  eret = pgaspi_dev_atomic_fetch_add(gctx,
				     segment_id, offset, rank,
				     val_add);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[GASPI_COLL_QP][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }

#ifdef GPI2_EXP_VERBS
    *val_old = swap_uint64((uint64_t)*((gaspi_atomic_value_t *) (gctx->nsrc.data.buf)));
#else
  *val_old = *((gaspi_atomic_value_t *) (gctx->nsrc.data.buf));
#endif

 endL:
  unlock_gaspi (&gctx->groups[0].gl);
  return eret;
}

#pragma weak gaspi_atomic_compare_swap = pgaspi_atomic_compare_swap
gaspi_return_t
pgaspi_atomic_compare_swap (const gaspi_segment_id_t segment_id,
			    const gaspi_offset_t offset,
			    const gaspi_rank_t rank,
			    const gaspi_atomic_value_t comparator,
			    const gaspi_atomic_value_t val_new,
			    gaspi_atomic_value_t * const val_old,
			    const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_atomic_compare_swap");
  gaspi_verify_remote_off(offset, segment_id, rank, sizeof(gaspi_atomic_value_t));
  gaspi_verify_null_ptr(val_old);
  gaspi_verify_unaligned_off(offset);

  gaspi_context_t * const gctx = &glb_gaspi_ctx;
  gaspi_return_t eret = GASPI_ERROR;

  if(lock_gaspi_tout (&gctx->groups[0].gl, timeout_ms))
    return GASPI_TIMEOUT;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if ( eret != GASPI_SUCCESS)
	{
	  goto endL;
	}
    }
  eret = pgaspi_dev_atomic_compare_swap(gctx,
					segment_id, offset, rank,
					comparator, val_new);

  if( eret != GASPI_SUCCESS )
    {
      gctx->state_vec[GASPI_COLL_QP][rank] = GASPI_STATE_CORRUPT;
      goto endL;
    }
#ifdef GPI2_EXP_VERBS
  *val_old = swap_uint64(*((gaspi_atomic_value_t *) (gctx->nsrc.data.buf)));
#else
  *val_old = *((gaspi_atomic_value_t *) (gctx->nsrc.data.buf));
#endif

 endL:
  unlock_gaspi (&gctx->groups[0].gl);
  return eret;
}
