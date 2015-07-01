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
#include "GPI2.h"
#include "GPI2_CM.h"
#include "GPI2_Dev.h"
#include "GPI2_SN.h"
#include "GPI2_Types.h"

gaspi_return_t
pgaspi_create_endpoint_to(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  const int i = (int) rank;

  if(lock_gaspi_tout(&gaspi_create_lock, timeout_ms))
    return GASPI_TIMEOUT;

  if(!glb_gaspi_ctx.ep_conn[i].istat)
    {
      if(pgaspi_dev_create_endpoint(i) < 0)
	{
	  unlock_gaspi(&gaspi_create_lock);
	  return GASPI_ERR_DEVICE;
	}
      glb_gaspi_ctx.ep_conn[i].istat = 1;
    }

  unlock_gaspi(&gaspi_create_lock);

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_connect_endpoint_to(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  const int i = (int) rank;
  gaspi_return_t eret = GASPI_ERROR;

  if(lock_gaspi_tout(&gaspi_ccontext_lock, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  /* already connected? */
  if(glb_gaspi_ctx.ep_conn[i].cstat)
    {
      eret = GASPI_SUCCESS;
    }
  else
    {
      if(pgaspi_dev_connect_context(i) != 0)
	{
	  eret = GASPI_ERR_DEVICE;
	}
      else
	{
	  glb_gaspi_ctx.ep_conn[i].cstat = 1;
	  eret = GASPI_SUCCESS;
	}
    }

  unlock_gaspi(&gaspi_ccontext_lock);

  return eret;
}

#pragma weak gaspi_connect = pgaspi_connect
gaspi_return_t
pgaspi_connect (const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;

  gaspi_verify_init("gaspi_connect");

  const int i = (int) rank;

  eret = pgaspi_create_endpoint_to(rank, timeout_ms);
  if( eret != GASPI_SUCCESS)
    {
      return eret;
    }

  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  if(glb_gaspi_ctx.ep_conn[i].cstat == 1)
    {
      /* already connected */
      unlock_gaspi(&glb_gaspi_ctx_lock);
      return GASPI_SUCCESS;
    }

  eret = gaspi_sn_command(GASPI_SN_CONNECT, rank, timeout_ms, NULL);
  if(eret != GASPI_SUCCESS)
    {
      if( GASPI_ERROR == eret)
	{
	  glb_gaspi_ctx.qp_state_vec[GASPI_SN][i] = GASPI_STATE_CORRUPT;
	}

      unlock_gaspi(&glb_gaspi_ctx_lock);
      return eret;
    }

  eret = pgaspi_connect_endpoint_to(rank, timeout_ms);

  unlock_gaspi(&glb_gaspi_ctx_lock);
  return eret;
}

#pragma weak gaspi_disconnect = pgaspi_disconnect
gaspi_return_t
pgaspi_disconnect(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;

  gaspi_verify_init("gaspi_disconnect");
  
  const int i = rank;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  /* Not connected? */
  if( 0 == glb_gaspi_ctx.ep_conn[i].cstat )
    {
      eret = GASPI_SUCCESS;
      goto errL;
    }
  
  eret = pgaspi_dev_disconnect_context(i);
  if(eret != GASPI_SUCCESS)
    goto errL;

  glb_gaspi_ctx.ep_conn[i].istat = 0;
  glb_gaspi_ctx.ep_conn[i].cstat = 0;

  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}
