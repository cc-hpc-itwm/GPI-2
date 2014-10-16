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

#include <sys/timeb.h>

#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_Utility.h"


#pragma weak gaspi_segment_max = pgaspi_segment_max
gaspi_return_t
pgaspi_segment_max (gaspi_number_t * const segment_max)
{
  gaspi_verify_null_ptr(segment_max);

  *segment_max = GASPI_MAX_MSEGS;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_size = pgaspi_segment_size
gaspi_return_t
pgaspi_segment_size (const gaspi_segment_id_t segment_id,
		     const gaspi_rank_t rank, gaspi_size_t * const size)
{
  gaspi_size_t seg_size = pgaspi_dev_get_mseg_size(segment_id, rank);
  
#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if (pgaspi_dev_get_rrmd(segment_id) == NULL)
    {
      gaspi_print_error("Invalid segment id");
      return GASPI_ERROR;
    }

  if (0 == seg_size)
    {
      gaspi_print_error("Invalid segment (size = 0)");
      return GASPI_ERROR;
    }

  gaspi_verify_null_ptr(size);

#endif
  
  *size = seg_size;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_ptr = pgaspi_segment_ptr
gaspi_return_t
pgaspi_segment_ptr (const gaspi_segment_id_t segment_id, gaspi_pointer_t * ptr)
{

#ifdef DEBUG  
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if (pgaspi_dev_get_rrmd(segment_id) == NULL)
    {
      gaspi_print_error("Invalid segment id");
      return GASPI_ERROR;
    }

  if (pgaspi_dev_get_mseg_size(segment_id, glb_gaspi_ctx.rank) == 0)
    {
      gaspi_print_error("Invalid segment (size = 0)");
      return GASPI_ERROR;
    }

  gaspi_verify_null_ptr(ptr);
#endif

  return pgaspi_dev_segment_ptr(segment_id, ptr);
}

#pragma weak gaspi_segment_list = pgaspi_segment_list
gaspi_return_t
pgaspi_segment_list (const gaspi_number_t num,
		    gaspi_segment_id_t * const segment_id_list)
{
  int i, idx = 0;

#ifdef DEBUG    
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
#endif
  
  //TODO: 256 -> readable
  for (i = 0; i < 256; i++)
    {
      if(pgaspi_dev_get_rrmd(i) != NULL)
	segment_id_list[idx++] = i;
    }

  if (idx != glb_gaspi_ctx.mseg_cnt)
    {
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_num = pgaspi_segment_num
gaspi_return_t
pgaspi_segment_num (gaspi_number_t * const segment_num)
{
  if (glb_gaspi_init)
    {
      *segment_num = glb_gaspi_ctx.mseg_cnt;
      return GASPI_SUCCESS;
    }

  gaspi_print_error("Invalid function before gaspi_proc_init");
  return GASPI_ERROR;
}

#pragma weak gaspi_segment_alloc = pgaspi_segment_alloc
gaspi_return_t
pgaspi_segment_alloc (const gaspi_segment_id_t segment_id,
		      const gaspi_size_t size,
		      const gaspi_alloc_t alloc_policy)
{

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  gaspi_return_t eret = GASPI_ERROR;
  
  lock_gaspi_tout (&gaspi_mseg_lock, GASPI_BLOCK);

  eret = pgaspi_dev_segment_alloc(segment_id, size, alloc_policy);
  
  unlock_gaspi (&gaspi_mseg_lock);

  return eret;
}

#pragma weak gaspi_segment_delete = pgaspi_segment_delete
gaspi_return_t
pgaspi_segment_delete (const gaspi_segment_id_t segment_id)
{

  if(!glb_gaspi_ib_init)
    {
      return GASPI_ERROR;
    }

#ifdef DEBUG  
  if (pgaspi_dev_get_rrmd(segment_id) == NULL)
    {
      gaspi_print_error("Invalid segment to delete");
      return GASPI_ERROR;
    }
  
  if (0 == pgaspi_dev_get_mseg_size(segment_id, glb_gaspi_ctx.rank))
    {
      gaspi_print_error("Invalid segment to delete");
      return GASPI_ERROR;
    }
  
#endif

  gaspi_return_t eret = GASPI_ERROR;
  
  lock_gaspi_tout(&gaspi_mseg_lock,GASPI_BLOCK);

  eret = pgaspi_dev_segment_delete(segment_id);

  unlock_gaspi (&gaspi_mseg_lock);

  return eret;
}


#pragma weak gaspi_segment_register = pgaspi_segment_register
gaspi_return_t
pgaspi_segment_register(const gaspi_segment_id_t segment_id,
			const gaspi_rank_t rank,
			const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG  
  if(!glb_gaspi_ib_init)
    return GASPI_ERROR;

  if(rank >= glb_gaspi_ctx.tnc || rank == glb_gaspi_ctx.rank)
    return GASPI_ERROR;

  if(pgaspi_dev_get_rrmd(segment_id) == NULL)
    return GASPI_ERROR;
  
  if(pgaspi_dev_get_mseg_size(segment_id, glb_gaspi_ctx.rank) == 0)
    return GASPI_ERROR;
#endif

  gaspi_return_t eret = GASPI_ERROR;

  if(lock_gaspi_tout(&glb_gaspi_ctx_lock, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_dev_segment_register(segment_id, rank, timeout_ms);

  unlock_gaspi(&glb_gaspi_ctx_lock);

  return eret;
}

#pragma weak gaspi_segment_create = pgaspi_segment_create
gaspi_return_t
pgaspi_segment_create(const gaspi_segment_id_t segment_id,
		      const gaspi_size_t size, const gaspi_group_t group,
		      const gaspi_timeout_t timeout_ms,
		      const gaspi_alloc_t alloc_policy)
{
#ifdef DEBUG  
  if(!glb_gaspi_ib_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if(group >= GASPI_MAX_GROUPS || glb_gaspi_group_ib[group].id < 0)
    {
      gaspi_print_error("Invalid group ( > GASPI_MAX_GROUPS || < 0)");
      return GASPI_ERROR;
    }
  
#endif
  
  if(pgaspi_dev_segment_alloc (segment_id, size, alloc_policy) != 0)
    {
      gaspi_print_error("Segment allocation failed");
      return GASPI_ERROR;
    }

  if(lock_gaspi_tout(&glb_gaspi_ctx_lock, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  gaspi_return_t eret = GASPI_ERROR;

  eret = pgaspi_dev_segment_register_group(segment_id, group, timeout_ms);
  if(eret != GASPI_SUCCESS)
    goto errL;
 
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;
  
errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}    
