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
#include <stdlib.h>

#include "GPI2.h"
#include "GPI2_OFI.h"

/* A range's registrations on every fabric. A segment inside another borrows
   them on fabrics using virtual addresses; refs counts owner and borrowers.
   Data registrations are listed in data_mrs. */
struct ofi_mr
{
  unsigned char* buf;
  size_t size;
  int refs;

  struct ofi_mr* covering;
  int borrowed[GPI2_OFI_MAX_FABRICS];
  struct ofi_mr* next;

  struct fid_mr* mr_fabric[GPI2_OFI_MAX_FABRICS];

  uint64_t rkey_fabric[GPI2_OFI_MAX_FABRICS];
};

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

void*
pgaspi_ofi_get_remote_mr_desc (void* mr)
{
  return fi_mr_desc (((struct ofi_mr*) mr)->mr_fabric[0]);
}

void*
pgaspi_ofi_get_local_mr_desc (void* mr)
{
  return fi_mr_desc (((struct ofi_mr*) mr)->mr_fabric[1]);
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

  /* With FI_MR_ENDPOINT: bind to the atomic endpoint, then enable. */
  if (fabric_ctx->info->domain_attr->mr_mode & FI_MR_ENDPOINT)
  {
    ret = fi_mr_bind (mr, &fabric_ctx->qAtomic->ep->fid, 0);
    if (0 == ret)
    {
      ret = fi_mr_enable (mr);
    }

    if (ret)
    {
      fprintf (stderr,
               "Failed to bind/enable memory region (ofi) (%d: %s).\n",
               ret, fi_strerror (-ret));

      fi_close (&mr->fid);
      return NULL;
    }
  }

  return mr;
}

/* Requested key (ignored with FI_MR_PROV_KEY); atomic: two locks register. */
static uint64_t
pgaspi_ofi_next_mr_key (void)
{
  static uint64_t next_key = 1;

  return __atomic_fetch_add (&next_key, 1, __ATOMIC_RELAXED);
}

static int
ofi_fabric_virt_addr (const struct ofi_fabric* const fabric_ctx)
{
  return 0 != (fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR);
}

/* Registers on every fabric; borrows covering's on fabrics with virtual
   addresses. */
static struct ofi_mr*
ofi_register_multi_domain_mr (gaspi_ofi_ctx* ofi_ctx,
                              unsigned char* buf,
                              size_t size,
                              struct ofi_mr* const covering)
{
  struct ofi_mr* ofi_mr = calloc (1, sizeof (struct ofi_mr));

  if (ofi_mr && ofi_ctx)
  {
    ofi_mr->buf = buf;
    ofi_mr->size = size;
    ofi_mr->refs = 1;

    for (int f = 0; f < GPI2_OFI_MAX_FABRICS; f++)
    {
      if (ofi_ctx->fabric_ctx[f] == NULL)
      {
        continue;
      }

      if (covering != NULL && ofi_fabric_virt_addr (ofi_ctx->fabric_ctx[f]))
      {
        ofi_mr->mr_fabric[f] = covering->mr_fabric[f];
        ofi_mr->rkey_fabric[f] = covering->rkey_fabric[f];
        ofi_mr->borrowed[f] = 1;
        ofi_mr->covering = covering;
        continue;
      }

      struct fid_mr* mr =
        ofi_register_memory_region (ofi_ctx->fabric_ctx[f],
                                    buf,
                                    size,
                                    pgaspi_ofi_next_mr_key());
      if (NULL == mr)
      {
        for (int g = 0; g < f; g++)
          if (ofi_mr->mr_fabric[g] && !ofi_mr->borrowed[g])
            fi_close (&ofi_mr->mr_fabric[g]->fid);

        free (ofi_mr);
        return NULL;
      }

      ofi_mr->mr_fabric[f] = mr;
      ofi_mr->rkey_fabric[f] = fi_mr_key (mr);
    }

    if (ofi_mr->covering != NULL)
    {
      ofi_mr->covering->refs++;
    }
  }

  return ofi_mr;
}

/* Drops a reference; the last one closes its own registrations and releases
   its covering one. Under data_mrs_lock. */
static int
ofi_release_mr (gaspi_ofi_ctx* const ofi_ctx, struct ofi_mr* const ofi_mr)
{
  if (--ofi_mr->refs > 0)
  {
    return 0;
  }

  int ret = 0;

  for (int f = 0; f < GPI2_OFI_MAX_FABRICS; f++)
  {
    if (ofi_mr->mr_fabric[f] && !ofi_mr->borrowed[f])
    {
      if (fi_close (&ofi_mr->mr_fabric[f]->fid))
        ret = -1;
    }
  }

  for (struct ofi_mr** p = &ofi_ctx->data_mrs; *p != NULL; p = &(*p)->next)
  {
    if (*p == ofi_mr)
    {
      *p = ofi_mr->next;
      break;
    }
  }

  if (ofi_mr->covering != NULL && ofi_release_mr (ofi_ctx, ofi_mr->covering))
  {
    ret = -1;
  }

  free (ofi_mr);

  return ret;
}

/* A live data registration covering the segment, or NULL: verbs;ofi_rxm
   fails a second registration of covered memory (the MR cache returns the
   same key, rxm refuses it). Only user-provided memory can lie inside a
   segment. Under data_mrs_lock. */
static struct ofi_mr*
ofi_covering_mr (gaspi_ofi_ctx* const ofi_ctx,
                 const gaspi_rc_mseg_t* const seg)
{
  if (!seg->user_provided)
  {
    return NULL;
  }

  for (struct ofi_mr* m = ofi_ctx->data_mrs; m != NULL; m = m->next)
  {
    if (m->buf <= seg->data.buf
        && seg->data.buf + seg->size <= m->buf + m->size)
    {
      return m;
    }
  }

  return NULL;
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

  if (seg->size > 0)
  {
    lock_gaspi (&ofi_ctx->data_mrs_lock);

    struct ofi_mr* const ofi_mr =
      ofi_register_multi_domain_mr (ofi_ctx,
                                    seg->data.buf,
                                    seg->size,
                                    ofi_covering_mr (ofi_ctx, seg));
    if (ofi_mr != NULL)
    {
      ofi_mr->next = ofi_ctx->data_mrs;
      ofi_ctx->data_mrs = ofi_mr;
    }

    unlock_gaspi (&ofi_ctx->data_mrs_lock);

    seg->mr[0] = ofi_mr;
    if (NULL == seg->mr[0])
    {
      return -1;
    }

    seg->rkey[0].local = ((struct ofi_mr*)seg->mr[0])->rkey_fabric[1];
    seg->rkey[0].remote = ((struct ofi_mr*)seg->mr[0])->rkey_fabric[0];
  }

  if (seg->notif_spc_size > 0)
  {
    seg->mr[1] = ofi_register_multi_domain_mr (ofi_ctx,
                                               seg->notif_spc.buf,
                                               seg->notif_spc_size,
                                               NULL);

    if (NULL == seg->mr[1])
    {
      return -1;
    }

    seg->rkey[1].local = ((struct ofi_mr*)seg->mr[1])->rkey_fabric[1];
    seg->rkey[1].remote = ((struct ofi_mr*)seg->mr[1])->rkey_fabric[0];
  }

  return 0;
}

int
pgaspi_dev_unregister_mem (gaspi_context_t const *const gctx,
                           gaspi_rc_mseg_t* seg)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;

  /* Note: Magic number 2 for the data and notifications spaces */
  for (int s = 0; s < 2; s++)
  {
    struct ofi_mr* ofi_mr = (struct ofi_mr*) seg->mr[s];
    if (!ofi_mr)
      continue;

    lock_gaspi (&ofi_ctx->data_mrs_lock);
    const int ret = ofi_release_mr (ofi_ctx, ofi_mr);
    unlock_gaspi (&ofi_ctx->data_mrs_lock);

    seg->mr[s] = NULL;

    if (ret)
      return -1;
  }

  return 0;
}
