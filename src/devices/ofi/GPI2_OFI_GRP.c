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

#include <stdio.h>

#include "GPI2.h"
#include "GPI2_GRP.h"
#include "GPI2_OFI.h"
#include "GPI2_Utility.h"

int
pgaspi_dev_poll_groups (gaspi_context_t* const gctx)
{
  int nelems = gctx->ne_count_grp;

  if (!nelems)
  {
    return 0;
  }

#define POLL_BATCH_SIZE 64
  struct fi_cq_err_entry comp[POLL_BATCH_SIZE];

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx = NULL;

  int ret;

  pgaspi_ofi_wait_begin();

  do
  {
    for (int fabric = 0; fabric < GPI2_OFI_MAX_FABRICS; fabric++)
    {
      fabric_ctx = ofi_ctx->fabric_ctx[fabric];

      if (fabric_ctx && fabric_ctx->qGroups)
      {
        int batch = nelems < POLL_BATCH_SIZE ? nelems : POLL_BATCH_SIZE;
        ret = fi_cq_read (fabric_ctx->qGroups->scq, comp, batch);
        if (ret > 0)
        {
          __atomic_sub_fetch (&gctx->ne_count_grp, ret, __ATOMIC_RELAXED);
        }
        else if (ret < 0 && ret != -FI_EAGAIN)
        {
          void* op_ctx = NULL;

          /* As with Infiniband, a failed write does not fail the collective,
             whose outcome are the sync flags: it completes and its rank is
             marked. */
          if (ret == -FI_EAVAIL &&
              0 == pgaspi_ofi_cq_readerr (fabric_ctx->qGroups->scq,
                                          &op_ctx))
          {
            __atomic_sub_fetch (&gctx->ne_count_grp, 1, __ATOMIC_RELAXED);

            /* The context is the target rank (PGASPI_OFI_RANK_CTX). */
            if (op_ctx != NULL)
            {
              gctx->state_vec[GASPI_COLL_QP][(uintptr_t) op_ctx - 1] =
                GASPI_STATE_CORRUPT;
            }
            continue;
          }

          GASPI_DEBUG_PRINT_ERROR
            ("Groups CQ read error (%d: %s)", ret, fi_strerror (-ret));

          pgaspi_ofi_wait_end();
          return -1;
        }
      }
    }
  } while (gctx->ne_count_grp > 0);

  pgaspi_ofi_wait_end();

  return 0;
}

/* Write to rank dst, with data as remote CQ data if with_data. */
static int
pgaspi_ofi_post_group_write (gaspi_context_t* const gctx,
                             void* local_addr,
                             int length,
                             int dst,
                             void* raddr,
                             unsigned char group,
                             const int with_data,
                             const uint64_t data)
{
 gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

 struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[dst];

 const gaspi_rc_mseg_t local_seg = gctx->groups[group].rrcd[gctx->rank];

 uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t) raddr :
    (uint64_t) ((unsigned char*) raddr -
                (unsigned char*) gctx->groups[group].rrcd[dst].data.buf);

  struct fid_ep *ep = fabric_ctx->qGroups->ep;
  const fi_addr_t dest =
    pgaspi_ofi_rma_target (fabric_ctx, fabric_ctx->groups_fi_addr[dst], dst);

  void* const desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], dst);

  uint64_t rkey =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, dst) ?
    gctx->groups[group].rrcd[dst].rkey[0].local :
    gctx->groups[group].rrcd[dst].rkey[0].remote;

  if (pgaspi_ofi_peer_known_dead (gctx, dst))
  {
    return -FI_EHOSTUNREACH;
  }

  int ret = 0;
  gaspi_cycles_t since = 0;
  do
  {
    ret = with_data
      ? fi_writedata (ep,
                      local_addr,
                      length,
                      desc,
                      data,
                      dest,
                      remote_addr,
                      rkey,
                      PGASPI_OFI_RANK_CTX (dst))
      : fi_write (ep,
                  local_addr,
                  length,
                  desc,
                  dest,
                  remote_addr,
                  rkey,
                  PGASPI_OFI_RANK_CTX (dst));

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating group write (%d: %s)\n", ret, fi_strerror (-ret));

      return ret;
    }

    if (ret)
    {
      /* The caller marks the peer and fails the collective. */
      if (pgaspi_ofi_peer_dead (gctx, dst, &since))
      {
        return -FI_EHOSTUNREACH;
      }

      pgaspi_ofi_progress_queue (fabric_ctx->qGroups);
    }
  } while (ret);

  __atomic_add_fetch (&gctx->ne_count_grp, 1, __ATOMIC_RELAXED);

  return 0;
}

int
pgaspi_dev_post_group_write (gaspi_context_t* const gctx,
                             void* local_addr,
                             int length,
                             int dst,
                             void* raddr,
                             unsigned char group)
{
  return pgaspi_ofi_post_group_write (gctx, local_addr, length, dst, raddr,
                                      group, 0, 0);
}

int
pgaspi_dev_post_group_write_sync (gaspi_context_t * const gctx,
                                  void *local_addr, int length, int dst,
                                  void *remote_addr,
                                  void *flag_local_addr,
                                  void *flag_remote_addr,
                                  uint32_t imm,
                                  unsigned char group)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;

  /* Relaxed ordering: the flag goes as remote CQ data with the data, and the
     receiver sets it on completion, never before the data (as
     WRITE_WITH_IMM). shm copies in order. */
  if (ofi_ctx->relaxed_ordering
      && ofi_ctx->rank_fabric_map[dst] == ofi_ctx->fabric_ctx[0])
  {
    return pgaspi_ofi_post_group_write (gctx, local_addr, length, dst,
                                        remote_addr, group, 1, imm) != 0;
  }

  if (pgaspi_dev_post_group_write (gctx, local_addr, length, dst,
                                   remote_addr, group) != 0)
  {
    return 1;
  }

  return pgaspi_dev_post_group_write (gctx, flag_local_addr, 1, dst,
                                      flag_remote_addr, group);
}

/* Relaxed ordering only (gaspi_run gives all ranks the setting). */
int
pgaspi_dev_poll_group_sync (gaspi_context_t * const gctx)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;

  if (!ofi_ctx->relaxed_ordering)
  {
    return 0;
  }

  struct fid_cq* const rcq = ofi_ctx->fabric_ctx[0]->qGroups->rcq;
  struct fi_cq_data_entry comp[POLL_BATCH_SIZE];

  const ssize_t ret = fi_cq_read (rcq, comp, POLL_BATCH_SIZE);

  if (ret == -FI_EAGAIN)
  {
    return 0;
  }

  /* A failed receive tells no flag: the missing one times out. */
  if (ret == -FI_EAVAIL)
  {
    pgaspi_ofi_cq_readerr (rcq, NULL);
    return 0;
  }

  if (ret < 0)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Groups receive CQ read error (%zd: %s)", ret, fi_strerror (-ret));
    return -1;
  }

  int nflags = 0;
  for (ssize_t i = 0; i < ret; i++)
  {
    if (comp[i].flags & FI_REMOTE_CQ_DATA)
    {
      pgaspi_group_sync_flag_set (gctx, (uint32_t) comp[i].data);
      nflags++;
    }
  }

  return nflags;
}
