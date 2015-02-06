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

#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_Utility.h"

#ifdef DEBUG  
static int
_check_func_params(const gaspi_segment_id_t segment_id_local,
		   const gaspi_offset_t offset_local,
		   const gaspi_size_t size)
{
  
  if (glb_gaspi_ctx.rrmd[segment_id_local] == NULL)
    {
      gaspi_print_error("Invalid local segment (gaspi_passive_receive)");    
      return -1;
    }
  
  if( offset_local > glb_gaspi_ctx.rrmd[segment_id_local][glb_gaspi_ctx.rank].size)
    {
      gaspi_print_error("Invalid offsets (gaspi_passive_receive)");    
      return -1;
    }
    
  if( size < 1 || size > GASPI_MAX_TSIZE_P )
    {
      gaspi_print_error("Invalid size (gaspi_passive_receive)");    
      return -1;
    }

  return 0;
}

#endif //DEBUG

#pragma weak gaspi_passive_transfer_size_max = pgaspi_passive_transfer_size_max
gaspi_return_t
pgaspi_passive_transfer_size_max (gaspi_size_t * const passive_transfer_size_max)
{

#ifdef DEBUG
  gaspi_verify_null_ptr(passive_transfer_size_max);
#endif  

  *passive_transfer_size_max = GASPI_MAX_TSIZE_P;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_passive_send     = pgaspi_passive_send
gaspi_return_t
pgaspi_passive_send (const gaspi_segment_id_t segment_id_local,
		    const gaspi_offset_t offset_local,
		    const gaspi_rank_t rank, const gaspi_size_t size,
		    const gaspi_timeout_t timeout_ms)
{
 
#ifdef DEBUG
  if(_check_func_params(segment_id_local, offset_local, size) < 0)
    {
      return GASPI_ERROR;
    }
#endif

  gaspi_return_t eret = GASPI_ERROR;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockPS, timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_passive_send(segment_id_local, offset_local, rank,
				 size, glb_gaspi_ctx.ne_count_p, timeout_ms);

  unlock_gaspi (&glb_gaspi_ctx.lockPS);

  return eret;
}

#pragma weak gaspi_passive_receive  = pgaspi_passive_receive
gaspi_return_t
pgaspi_passive_receive (const gaspi_segment_id_t segment_id_local,
		       const gaspi_offset_t offset_local,
		       gaspi_rank_t * const rem_rank, const gaspi_size_t size,
		       const gaspi_timeout_t timeout_ms)
{
  
#ifdef DEBUG
  if(_check_func_params(segment_id_local, offset_local, size) < 0)
    {
      return GASPI_ERROR;
    }
#endif

  gaspi_return_t eret = GASPI_ERROR;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx.lockPR, timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_passive_receive(segment_id_local, offset_local, rem_rank,
				    size, timeout_ms);

  unlock_gaspi (&glb_gaspi_ctx.lockPR);

  return eret;
}
