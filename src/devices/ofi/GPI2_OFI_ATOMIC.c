/*
  Copyright (c) Fraunhofer ITWM, 2013-2025

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

  // Wait for the operation to complete by polling the CQ
  do
  {
    struct fi_cq_data_entry cq_entry;
    ret = fi_cq_read (fabric_ctx->qAtomic->scq, &cq_entry, 1);

    if (ret < 0 && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Atomic CQ read error (%ld: %s)\n", ret, fi_strerror (-ret));

      return GASPI_ERR_DEVICE;
    }
  }
  while (ret <= 0);

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

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  //TODO: do this once, somewhere else
  size_t count = 0;
  if (fi_fetch_atomicvalid (fabric_ctx->qAtomic->ep,
                            FI_UINT64,
                            FI_SUM,
                            &count)
      != 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Atomic operation not supported.");
    return GASPI_ERR_DEVICE;
  }


  uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t)(gctx->rrmd[segment_id][rank].data.addr + offset) :
    (uint64_t) offset;

  void* const local_desc =
    pgaspi_dev_get_mr_desc (gctx,
                            gctx->rrmd[segment_id][gctx->rank].mr[0],
                            rank);
  void* const result_desc = pgaspi_dev_get_mr_desc (gctx, gctx->nsrc.mr[0], rank);

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    gctx->rrmd[segment_id][rank].rkey[0].local :
    gctx->rrmd[segment_id][rank].rkey[0].remote;

  ssize_t ret;

  do
  {
    ret = fi_fetch_atomic (fabric_ctx->qAtomic->ep,
                           &val_add,
                           1,
                           local_desc,
                           gctx->nsrc.data.buf,
                           result_desc,
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

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  ssize_t ret;

  uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t)(gctx->rrmd[segment_id][rank].data.addr + offset) :
    (uint64_t) offset;

    void* const local_desc =
    pgaspi_dev_get_mr_desc (gctx,
                            gctx->rrmd[segment_id][gctx->rank].mr[0],
                            rank);
  void* const result_desc =
    pgaspi_dev_get_mr_desc (gctx, gctx->nsrc.mr[0], rank);

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    gctx->rrmd[segment_id][rank].rkey[0].local :
    gctx->rrmd[segment_id][rank].rkey[0].remote;

  do
  {
    ret = fi_compare_atomic (fabric_ctx->qAtomic->ep,
                             &val_new,
                             1,
                             local_desc,
                             &comparator,
                             local_desc,
                             gctx->nsrc.data.buf,
                             result_desc,
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
  } while (ret);

  return pgaspi_dev_poll_single_entry (fabric_ctx);
}
