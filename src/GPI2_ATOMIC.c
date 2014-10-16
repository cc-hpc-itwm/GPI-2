/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2014

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

#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_Utility.h"

#pragma weak gaspi_atomic_fetch_add = pgaspi_atomic_fetch_add
gaspi_return_t
pgaspi_atomic_fetch_add (const gaspi_segment_id_t segment_id,
			 const gaspi_offset_t offset, const gaspi_rank_t rank,
			 const gaspi_atomic_value_t val_add,
			 gaspi_atomic_value_t * const val_old,
			 const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (pgaspi_dev_get_rrmd(segment_id) == NULL)
    {
      gaspi_print_error("Invalid segment (gaspi_atomic_fetch_add)");    
      return GASPI_ERROR;
    }
  
  if( rank >= glb_gaspi_ctx.tnc)
    {
      gaspi_print_error("Invalid rank (gaspi_atomic_fetch_add)");    
      return GASPI_ERROR;
    }
  
  if( offset > pgaspi_dev_get_mseg_size(segment_id, rank))
    {
      gaspi_print_error("Invalid offsets (gaspi_atomic_fetch_add)");    
      return GASPI_ERROR;
    }

  if( val_old == NULL)
    {
      gaspi_print_error("Invalid pointer in parameter val_old (gaspi_atomic_fetch_add)");    
      return GASPI_ERROR;
    }

  if(timeout_ms < GASPI_TEST || timeout_ms > GASPI_BLOCK)
    {
      gaspi_print_error("Invalid timeout: %lu", timeout_ms);
      return GASPI_ERROR;
    }

#endif
  gaspi_return_t eret = GASPI_ERROR;
  
  if (offset & 0x7)
    {
      gaspi_print_error("Unaligned offset for atomic operation (fetch_add)");
      return GASPI_ERROR;
    }
  
  if(lock_gaspi_tout (&glb_gaspi_group_ib[0].gl, timeout_ms))
    return GASPI_TIMEOUT;
  
  eret = pgaspi_dev_atomic_fetch_add(segment_id, offset, rank,
				     val_add, val_old, timeout_ms);
  
  unlock_gaspi (&glb_gaspi_group_ib[0].gl);
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
#ifdef DEBUG
  if (pgaspi_dev_get_rrmd(segment_id) == NULL)
    {
      gaspi_print_error("Invalid segment (gaspi_atomic_compare_swap)");    
      return GASPI_ERROR;
    }
  
  if( rank >= glb_gaspi_ctx.tnc)
    {
      gaspi_print_error("Invalid rank (gaspi_atomic_compare_swap)");    
      return GASPI_ERROR;
    }

  if( offset > pgaspi_dev_get_mseg_size(segment_id, rank))
    {
      gaspi_print_error("Invalid offsets (gaspi_atomic_compare_swap)");    
      return GASPI_ERROR;
    }

  if( val_old == NULL)
    {
      gaspi_print_error("Invalid pointer in parameter val_old (gaspi_atomic_compare_swap)");    
      return GASPI_ERROR;
    }
  
  if(timeout_ms < GASPI_TEST || timeout_ms > GASPI_BLOCK)
    {
      gaspi_print_error("Invalid timeout: %lu", timeout_ms);
      return GASPI_ERROR;
    }
#endif

  gaspi_return_t eret = GASPI_ERROR;
  
  if (offset & 0x7)
    {
      gaspi_print_error("Unaligned offset for atomic operation (compare_swap");
      return GASPI_ERROR;
    }
  
  if(lock_gaspi_tout (&glb_gaspi_group_ib[0].gl, timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_atomic_compare_swap(segment_id, offset, rank,
					comparator, val_new, val_old,
					timeout_ms);

  unlock_gaspi (&glb_gaspi_group_ib[0].gl);

  return eret;
}

