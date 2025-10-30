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
#include <stdlib.h>

#include "GPI2_OFI.h"

#define MR_KEY 42//0xC0DE //TODO: new name, new mechanism?

struct ofi_mr
{
  /* void* buffer; */
  /* size_t size; */

  struct fid_mr* mr_fabric[GPI2_OFI_MAX_FABRICS];

  uint64_t rkey_fabric[GPI2_OFI_MAX_FABRICS];
};

uint64_t
pgaspi_dev_get_mr_rkey (gaspi_context_t const *const gctx,
                        void* mr,
                        gaspi_rank_t rank)
{
  if (gctx && gctx->device)
  {
    gaspi_ofi_ctx* const ofi_ctx = (gaspi_ofi_ctx*) gctx->device->ctx;

    struct ofi_mr* ofi_mr = (struct ofi_mr*) mr;

    if (ofi_ctx && ofi_mr)
    {
      if (pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank))
      {
        return ofi_mr->rkey_fabric[1];
      }

      return ofi_mr->rkey_fabric[0];
    }
  }

  OFI_UNREACHABLE();

  return 0;
}

void*
pgaspi_dev_get_mr_desc (gaspi_context_t const *const gctx,
                        void* mr,
                        gaspi_rank_t rank)
{
  if (gctx && gctx->device)
  {
    gaspi_ofi_ctx* const ofi_ctx = (gaspi_ofi_ctx*) gctx->device->ctx;

    struct ofi_mr* ofi_mr = (struct ofi_mr*) mr;

    if (ofi_ctx && ofi_mr)
    {
      if (pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank))

      {
        return fi_mr_desc (ofi_mr->mr_fabric[1]);
      }

      return fi_mr_desc (ofi_mr->mr_fabric[0]);
    }
  }

  OFI_UNREACHABLE();

  return NULL;
}

static struct fid_mr*
ofi_register_memory_region (struct ofi_fabric* fabric_ctx,
                            unsigned char* buf,
                            size_t size,
                            uint64_t key)
{
  struct fid_mr *mr = NULL;
  int ret = fi_mr_reg (fabric_ctx->domain,
                       buf,
                       size,
                       FI_SEND    | FI_RECV |
                         FI_WRITE | FI_REMOTE_WRITE |
                         FI_READ  | FI_REMOTE_READ,
                       0,
                       key,
                       0,
                       &mr,
                       NULL);
  if (ret)
  {
    fprintf (stderr,
             "Failed to register memory region (ofi) (%d: %s).\n",
             ret, fi_strerror (-ret));

    return NULL;
  }

  return mr;
}

static struct ofi_mr*
ofi_register_multi_domain_mr (gaspi_ofi_ctx* ofi_ctx,
                              unsigned char* buf,
                              size_t size,
                              uint64_t key)
{
  struct ofi_mr* ofi_mr = calloc (1, sizeof (struct ofi_mr));

  if (ofi_mr && ofi_ctx && ofi_ctx->fabric_ctx)
  {
    for (int f = 0; f < GPI2_OFI_MAX_FABRICS; f++)
    {
      if (ofi_ctx->fabric_ctx[f] != NULL)
      {
        struct fid_mr* mr =
          ofi_register_memory_region (ofi_ctx->fabric_ctx[f],
                                      buf,
                                      size,
                                      key);
        ofi_mr->mr_fabric[f] = mr;
        ofi_mr->rkey_fabric[f] = mr ? fi_mr_key (mr) : 0;
      }
    }
  }

  return ofi_mr;
}

int
pgaspi_dev_register_mem (gaspi_context_t const *const gctx,
                         gaspi_rc_mseg_t* seg)
{
  if (NULL == gctx || NULL == seg)
  {
    return -1;
  }

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  static uint64_t key = MR_KEY;

  if (seg->size > 0)
  {
    seg->mr[0] = ofi_register_multi_domain_mr (ofi_ctx,
                                               seg->data.buf,
                                               seg->size,
                                               key);
    if (NULL == seg->mr[0])
    {
      return -1;
    }

    seg->rkey[0].local = ((struct ofi_mr*)seg->mr[0])->rkey_fabric[1];
    seg->rkey[0].remote = ((struct ofi_mr*)seg->mr[0])->rkey_fabric[0];

    //TODO: we need a more elegant approach
    key++;
  }

  if (seg->notif_spc_size > 0)
  {
    seg->mr[1] = ofi_register_multi_domain_mr (ofi_ctx,
                                               seg->notif_spc.buf,
                                               seg->notif_spc_size,
                                               key);

    if (NULL == seg->mr[1])
    {
      return -1;
    }

    seg->rkey[1].local = ((struct ofi_mr*)seg->mr[1])->rkey_fabric[1];
    seg->rkey[1].remote = ((struct ofi_mr*)seg->mr[1])->rkey_fabric[0];

    //TODO: we need a more elegant approach
    key++;
  }

  return 0;
}

int
pgaspi_dev_unregister_mem (gaspi_context_t const *const gctx,
                           gaspi_rc_mseg_t* seg)
{
  /* Note: Magic number 2 for the data and notifications spaces */
  for (int s = 0; s < 2; s++)
  {
    for (int f = 0; f < GPI2_OFI_MAX_FABRICS; f++)
    {
      struct ofi_mr* ofi_mr = (struct ofi_mr*) seg->mr[s];

      if (ofi_mr && ofi_mr->mr_fabric[f])
      {
        if (fi_close (&((struct fid_mr*) ofi_mr->mr_fabric[f])->fid))
        {
          return -1;
        }
      }
    }
  }

  return 0;
}
