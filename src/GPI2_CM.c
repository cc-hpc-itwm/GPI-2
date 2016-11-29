/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2016

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
  gaspi_context_t * const gctx = &glb_gaspi_ctx;
  const int i = (int) rank;

  if( lock_gaspi_tout(&(gctx->create_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  if( GASPI_ENDPOINT_NOT_CREATED == gctx->ep_conn[i].istat )
    {
      if( pgaspi_dev_create_endpoint(i) < 0 )
	{
	  unlock_gaspi(&(gctx->create_lock));
	  return GASPI_ERR_DEVICE;
	}
      gctx->ep_conn[i].istat = GASPI_ENDPOINT_CREATED;
    }

  unlock_gaspi(&(gctx->create_lock));

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_connect_endpoint_to(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  const int i = (int) rank;
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( lock_gaspi_tout(&(gctx->ccontext_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  if( GASPI_ENDPOINT_CONNECTED == gctx->ep_conn[i].cstat )
    {
      eret = GASPI_SUCCESS;
    }
  else
    {
      if( pgaspi_dev_connect_context(i) != 0 )
	{
	  eret = GASPI_ERR_DEVICE;
	}
      else
	{
	  gctx->ep_conn[i].cstat = GASPI_ENDPOINT_CONNECTED;
	  eret = GASPI_SUCCESS;
	}
    }

  unlock_gaspi(&(gctx->ccontext_lock));

  return eret;
}

#pragma weak gaspi_connect = pgaspi_connect
gaspi_return_t
pgaspi_connect (const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_connect");

  const int i = (int) rank;

  eret = pgaspi_create_endpoint_to(rank, timeout_ms);
  if( eret != GASPI_SUCCESS)
    {
      return eret;
    }

  if( lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  if( GASPI_ENDPOINT_CONNECTED == gctx->ep_conn[i].cstat )
    {
      unlock_gaspi(&(gctx->ctx_lock));
      return GASPI_SUCCESS;
    }

  eret = gaspi_sn_command(GASPI_SN_CONNECT, rank, timeout_ms, NULL);
  if( eret != GASPI_SUCCESS )
    {
      if( GASPI_ERROR == eret)
	{
	  gctx->qp_state_vec[GASPI_SN][i] = GASPI_STATE_CORRUPT;
	}

      unlock_gaspi(&(gctx->ctx_lock));
      return eret;
    }

  eret = pgaspi_connect_endpoint_to(rank, timeout_ms);

  unlock_gaspi(&(gctx->ctx_lock));
  return eret;
}

gaspi_return_t
pgaspi_local_disconnect(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  const int i = rank;
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[i].cstat )
    {
      unlock_gaspi(&(gctx->ctx_lock));
      return GASPI_SUCCESS;
    }

  eret = pgaspi_dev_disconnect_context(i);
  if( eret == GASPI_SUCCESS)
    {
      gctx->ep_conn[i].istat = GASPI_ENDPOINT_NOT_CREATED;
      gctx->ep_conn[i].cstat = GASPI_ENDPOINT_DISCONNECTED;
    }

  unlock_gaspi(&(gctx->ctx_lock));
  return eret;
}

#pragma weak gaspi_disconnect = pgaspi_disconnect
gaspi_return_t
pgaspi_disconnect(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_disconnect");

  const int i = rank;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[i].cstat )
    {
      return GASPI_SUCCESS;
    }

  eret = pgaspi_local_disconnect(rank, timeout_ms);
  if( eret != GASPI_SUCCESS )
    {
      return eret;
    }

  if( lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  /* we can still get trapped inside the sn command, trying to connect */
  /* to a remote rank and in case the remote rank is */
  /* finished/gone. Thus we go with GASPI_TEST as timeout for now.  */
  eret = gaspi_sn_command(GASPI_SN_DISCONNECT, rank, GASPI_TEST, NULL);

  unlock_gaspi (&(gctx->ctx_lock));
  return eret;
}
