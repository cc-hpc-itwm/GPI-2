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

#include <stdio.h>

#include "GPI2.h"
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

  struct fi_cq_err_entry* comp =
    malloc (nelems * sizeof (struct fi_cq_err_entry));
  if (NULL == comp)
  {
    return GASPI_ERR_MEMALLOC;
  }

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx = NULL;

  int ret;

  do
  {
    for (int fabric = 0; fabric < GPI2_OFI_MAX_FABRICS; fabric++)
    {
      fabric_ctx = ofi_ctx->fabric_ctx[fabric];

      if (fabric_ctx && fabric_ctx->qGroups)
      {
        ret = fi_cq_read (fabric_ctx->qGroups->scq, comp, nelems);
        if (ret > 0)
        {
          __atomic_sub_fetch (&gctx->ne_count_grp, ret, __ATOMIC_RELAXED);
        }
      }
    }
  } while (gctx->ne_count_grp > 0);

  return 0;
}

int
pgaspi_dev_post_group_write (gaspi_context_t* const gctx,
                             void* local_addr,
                             int length, //TODO: size_t
                             int dst,
                             void* raddr,
                             unsigned char group)
{
 gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

 struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[dst];

 const gaspi_rc_mseg_t local_seg = gctx->groups[group].rrcd[gctx->rank];
// const gaspi_rc_mseg_t remote_seg = gctx->groups[group].rrcd[dst];

 uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t) raddr :
    (uint64_t) ((unsigned char*) raddr -
                (unsigned char*) gctx->groups[group].rrcd[dst].data.buf);

  struct fid_ep *ep = fabric_ctx->qGroups->ep;
  const fi_addr_t dest = fabric_ctx->groups_fi_addr[dst];

  void* const desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], dst);

  uint64_t rkey =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, dst) ?
    gctx->groups[group].rrcd[dst].rkey[0].local :
    gctx->groups[group].rrcd[dst].rkey[0].remote;

  int ret = 0;
  do
  {
    ret = fi_write (ep,
                    local_addr,
                    length,
                    desc,
                    dest,
                    remote_addr,
                    rkey,
                    NULL);

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating group write (%d: %s)\n", ret, fi_strerror (-ret));

      return ret;
    }
  } while (ret);

  __atomic_add_fetch (&gctx->ne_count_grp, 1, __ATOMIC_RELAXED);

  return 0;
}
