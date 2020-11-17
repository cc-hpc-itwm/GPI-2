/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2020

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

#pragma weak gaspi_passive_transfer_size_min = pgaspi_passive_transfer_size_min
gaspi_return_t
pgaspi_passive_transfer_size_min (gaspi_size_t *
                                  const passive_transfer_size_min)
{
  GASPI_VERIFY_NULL_PTR (passive_transfer_size_min);

  *passive_transfer_size_min = GASPI_MIN_TSIZE_P;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_passive_transfer_size_max = pgaspi_passive_transfer_size_max
gaspi_return_t
pgaspi_passive_transfer_size_max (gaspi_size_t *
                                  const passive_transfer_size_max)
{
  GASPI_VERIFY_NULL_PTR (passive_transfer_size_max);

  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  *passive_transfer_size_max = gctx->config->passive_transfer_size_max;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_passive_send = pgaspi_passive_send
gaspi_return_t
pgaspi_passive_send (const gaspi_segment_id_t segment_id_local,
                     const gaspi_offset_t offset_local,
                     const gaspi_rank_t rank,
                     const gaspi_size_t size,
                     const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_passive_send");
  GASPI_VERIFY_LOCAL_OFF (offset_local, segment_id_local, size);
  GASPI_VERIFY_COMM_SIZE (size,
                          segment_id_local,
                          segment_id_local,
                          gctx->rank, GASPI_MIN_TSIZE_P,
                          gctx->config->passive_transfer_size_max);
  GASPI_VERIFY_RANK (rank);

  gaspi_return_t eret = GASPI_ERROR;

  if (lock_gaspi_tout (&gctx->lockPS, timeout_ms))
  {
    return GASPI_TIMEOUT;
  }

  if (GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat)
  {
    eret = pgaspi_connect ((gaspi_rank_t) rank, timeout_ms);
    if (eret != GASPI_SUCCESS)
    {
      goto endL;
    }
  }

  eret = pgaspi_dev_passive_send (gctx,
                                  segment_id_local, offset_local, rank,
                                  size, timeout_ms);

  if (eret == GASPI_ERROR)
  {
    gctx->state_vec[GASPI_PASSIVE_QP][rank] = GASPI_STATE_CORRUPT;
  }

endL:
  unlock_gaspi (&gctx->lockPS);
  return eret;
}

#pragma weak gaspi_passive_receive = pgaspi_passive_receive
gaspi_return_t
pgaspi_passive_receive (const gaspi_segment_id_t segment_id_local,
                        const gaspi_offset_t offset_local,
                        gaspi_rank_t * const rem_rank,
                        const gaspi_size_t size,
                        const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_passive_receive");
  GASPI_VERIFY_LOCAL_OFF (offset_local, segment_id_local, size);
  GASPI_VERIFY_COMM_SIZE (size,
                          segment_id_local,
                          segment_id_local,
                          gctx->rank, GASPI_MIN_TSIZE_P,
                          gctx->config->passive_transfer_size_max);

  if (lock_gaspi_tout (&gctx->lockPR, timeout_ms))
  {
    return GASPI_TIMEOUT;
  }

  gaspi_return_t const eret =
    pgaspi_dev_passive_receive (gctx,
                                segment_id_local, offset_local, rem_rank,
                                size, timeout_ms);

  unlock_gaspi (&gctx->lockPR);

  return eret;
}
