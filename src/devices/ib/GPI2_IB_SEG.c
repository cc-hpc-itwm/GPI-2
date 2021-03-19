/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2021

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

#include <errno.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <unistd.h>

#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_IB.h"

int
pgaspi_dev_register_mem (gaspi_context_t const *const gctx,
                         gaspi_rc_mseg_t * seg)
{
  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  seg->mr[0] = ibv_reg_mr (ib_dev_ctx->pd,
                           seg->data.buf,
                           seg->size,
                           IBV_ACCESS_REMOTE_WRITE
                           | IBV_ACCESS_LOCAL_WRITE
                           | IBV_ACCESS_REMOTE_READ
                           | IBV_ACCESS_REMOTE_ATOMIC);

  if (seg->mr[0] == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Memory registration failed (libibverbs)");
    return -1;
  }

  seg->rkey[0] = ((struct ibv_mr *) seg->mr[0])->rkey;

  if (seg->notif_spc.buf != NULL)
  {
    seg->mr[1] = ibv_reg_mr (ib_dev_ctx->pd,
                             seg->notif_spc.buf,
                             seg->notif_spc_size,
                             IBV_ACCESS_REMOTE_WRITE
                             | IBV_ACCESS_LOCAL_WRITE
                             | IBV_ACCESS_REMOTE_READ
                             | IBV_ACCESS_REMOTE_ATOMIC);

    if (seg->mr[1] == NULL)
    {
      GASPI_DEBUG_PRINT_ERROR ("Memory registration failed (libibverbs)");
      return -1;
    }

    seg->rkey[1] = ((struct ibv_mr *) seg->mr[1])->rkey;
  }

  return 0;
}

int
pgaspi_dev_unregister_mem (gaspi_context_t const * const GASPI_UNUSED (gctx),
                           gaspi_rc_mseg_t * seg)
{
  if (seg->mr[0] != NULL)
  {

    if (ibv_dereg_mr ((struct ibv_mr *) seg->mr[0]))
    {
      GASPI_DEBUG_PRINT_ERROR ("Memory de-registration failed (libibverbs)");
      return -1;
    }
  }

  if (seg->mr[1] != NULL)
  {
    if (ibv_dereg_mr ((struct ibv_mr *) seg->mr[1]))
    {
      GASPI_DEBUG_PRINT_ERROR ("Memory de-registration failed (libibverbs)");
      return -1;
    }
  }

  return 0;
}
