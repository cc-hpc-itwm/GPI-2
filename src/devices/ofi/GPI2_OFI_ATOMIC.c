/*
  Copyright (c) Fraunhofer ITWM, 2013-2026

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

#include <rdma/fi_atomic.h>

#include "GPI2.h"
#include "GPI2_OFI.h"
#include "GPI2_Utility.h"

//TODO: merge scattered polling functions
static inline gaspi_return_t
pgaspi_dev_poll_single_entry (struct ofi_fabric* fabric_ctx)
{
  ssize_t ret = -1;

  pgaspi_ofi_wait_begin();

  // Wait for the operation to complete by polling the CQ
  do
  {
    struct fi_cq_data_entry cq_entry;
    ret = fi_cq_read (fabric_ctx->qAtomic->scq, &cq_entry, 1);

    if (ret < 0 && ret != -FI_EAGAIN)
    {
      pgaspi_ofi_wait_end();

      /* An error completion fails the operation; the caller marks the rank. */
      if (ret == -FI_EAVAIL
          && 0 == pgaspi_ofi_cq_readerr (fabric_ctx->qAtomic->scq, NULL))
      {
        return GASPI_ERROR;
      }

      GASPI_DEBUG_PRINT_ERROR
        ("Atomic CQ read error (%ld: %s)\n", ret, fi_strerror (-ret));

      return GASPI_ERR_DEVICE;
    }
  }
  while (ret <= 0);

  pgaspi_ofi_wait_end();

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_atomic_fetch_add (gaspi_context_t * const gctx,
                             const gaspi_segment_id_t segment_id,
                             const gaspi_offset_t offset,
                             const gaspi_rank_t rank,
                             const gaspi_atomic_value_t val_add)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  /* Always the remote fabric: atomics across providers are not atomic. */
  struct ofi_fabric* fabric_ctx = ofi_ctx->fabric_ctx[0];

  uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t)(gctx->rrmd[segment_id][rank].data.addr + offset) :
    (uint64_t) offset;

  /* The operand goes into registered internal memory (FI_MR_LOCAL needs a
     descriptor); others get none (cxi maps it). Atomics are serialized and
     complete before returning. */
  gaspi_atomic_value_t* const values =
    (gaspi_atomic_value_t*) gctx->nsrc.data.buf;
  values[1] = val_add;
  void* const desc = pgaspi_ofi_get_remote_mr_desc (gctx->nsrc.mr[0]);
  void* const operand_desc =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_LOCAL ? desc : NULL;

  uint64_t key = gctx->rrmd[segment_id][rank].rkey[0].remote;

  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    return GASPI_ERROR;
  }

  ssize_t ret;
  gaspi_cycles_t since = 0;

  do
  {
    ret = fi_fetch_atomic (fabric_ctx->qAtomic->ep,
                           &values[1],
                           1,
                           operand_desc,
                           &values[0],
                           desc,
                           fabric_ctx->atomic_fi_addr[rank],
                           remote_addr,
                           key,
                           FI_UINT64,
                           FI_SUM,
                           NULL);

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR ("Error initiating atomic fetch_add");
      return GASPI_ERR_DEVICE;
    }

    if (ret)
    {
      /* Marked dead by the ping. */
      if (pgaspi_ofi_peer_dead (gctx, rank, &since))
      {
        return GASPI_ERROR;
      }

      pgaspi_ofi_progress_queue (fabric_ctx->qAtomic);
    }
  } while (ret);

  return pgaspi_dev_poll_single_entry (fabric_ctx);
}

gaspi_return_t
pgaspi_dev_atomic_compare_swap (gaspi_context_t * const gctx,
                                const gaspi_segment_id_t segment_id,
                                const gaspi_offset_t offset,
                                const gaspi_rank_t rank,
                                const gaspi_atomic_value_t comparator,
                                const gaspi_atomic_value_t val_new)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  /* Always the remote fabric: atomics across providers are not atomic. */
  struct ofi_fabric* fabric_ctx = ofi_ctx->fabric_ctx[0];

  ssize_t ret;

  uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t)(gctx->rrmd[segment_id][rank].data.addr + offset) :
    (uint64_t) offset;

  /* The operands go into internal registered memory, as for fetch_add. */
  gaspi_atomic_value_t* const values =
    (gaspi_atomic_value_t*) gctx->nsrc.data.buf;
  values[1] = val_new;
  values[2] = comparator;
  void* const desc = pgaspi_ofi_get_remote_mr_desc (gctx->nsrc.mr[0]);
  void* const operand_desc =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_LOCAL ? desc : NULL;

  uint64_t key = gctx->rrmd[segment_id][rank].rkey[0].remote;

  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    return GASPI_ERROR;
  }

  gaspi_cycles_t since = 0;
  do
  {
    ret = fi_compare_atomic (fabric_ctx->qAtomic->ep,
                             &values[1],
                             1,
                             operand_desc,
                             &values[2],
                             operand_desc,
                             &values[0],
                             desc,
                             fabric_ctx->atomic_fi_addr[rank],
                             remote_addr,
                             key,
                             FI_UINT64,
                             FI_CSWAP,
                             NULL);

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR ("Error initiating atomic compare_swap");
      return GASPI_ERR_DEVICE;
    }

    if (ret)
    {
      /* Marked dead by the ping. */
      if (pgaspi_ofi_peer_dead (gctx, rank, &since))
      {
        return GASPI_ERROR;
      }

      pgaspi_ofi_progress_queue (fabric_ctx->qAtomic);
    }
  } while (ret);

  return pgaspi_dev_poll_single_entry (fabric_ctx);
}
