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

#include "GPI2.h"
#include "GPI2_OFI.h"
#include "GPI2_Utility.h"

//TODO: there is some repetition (polling and posting requests) that
//could be improved

static gaspi_return_t
ofi_poll_cq_with_timeout (gaspi_context_t * const gctx,
                          struct fid_cq* const cq,
                          struct fi_cq_data_entry* cq_entry,
                          const gaspi_timeout_t timeout_ms)
{
  const gaspi_cycles_t s0 = gaspi_get_cycles();

  int ret;
  do
  {
    ret = fi_cq_read (cq, cq_entry, 1);
    if (ret < 0 && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Passive CQ read error (%d: %s)\n", ret, fi_strerror (-ret));
      return GASPI_ERR_DEVICE;
    }

    if (ret == -FI_EAGAIN)
    {
      const gaspi_cycles_t s1 = gaspi_get_cycles();
      const gaspi_cycles_t tdelta = s1 - s0;

      const float ms = (float) tdelta * gctx->cycles_to_msecs;

      if (ms > timeout_ms)
      {
        return GASPI_TIMEOUT;
      }
    }
  } while (ret != 1);

  return GASPI_SUCCESS;
}


static gaspi_return_t
ofi_poll_cqs_with_timeout (gaspi_context_t * const gctx,
                           struct fid_cq* const cq1,
                           struct fid_cq* const cq2,
                           struct fi_cq_data_entry* cq_entry,
                           const gaspi_timeout_t timeout_ms)
{
  const gaspi_cycles_t s0 = gaspi_get_cycles();

  int ret;
  do
  {
    ret = fi_cq_read (cq1, cq_entry, 1);
    if (ret < 0 && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Passive CQ read error (%d: %s)\n", ret, fi_strerror (-ret));
      return GASPI_ERR_DEVICE;
    }

    if (ret == 1)
    {
      break;
    }


    ret = fi_cq_read (cq2, cq_entry, 1);
    if (ret < 0 && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Passive CQ read error (%d: %s)\n", ret, fi_strerror (-ret));
      return GASPI_ERR_DEVICE;
    }

    if (ret == -FI_EAGAIN)
    {
      const gaspi_cycles_t s1 = gaspi_get_cycles();
      const gaspi_cycles_t tdelta = s1 - s0;

      const float ms = (float) tdelta * gctx->cycles_to_msecs;

      if (ms > timeout_ms)
      {
        return GASPI_TIMEOUT;
      }
    }
    if (ret == 1)
    {
      break;
    }
  } while (ret != 1);

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_passive_send (gaspi_context_t * const gctx,
                         const gaspi_segment_id_t segment_id_local,
                         const gaspi_offset_t offset_local,
                         const gaspi_rank_t rank,
                         const gaspi_size_t size,
                         const gaspi_timeout_t timeout_ms)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];

  void* const local_desc =
    pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], gctx->rank);

  int ret;

  do
  {

    ret = fi_senddata (fabric_ctx->qP->ep,
                       (void*)local_seg.data.addr + offset_local,
                       size,
                       local_desc,
                       gctx->rank,
                       fabric_ctx->passive_fi_addr[rank],
                       NULL);
    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating passive send (%d: %s)\n",
         ret, fi_strerror (-ret));

      return GASPI_ERR_DEVICE;
    }
  } while (ret);

  // Wait for operation to complete
  struct fi_cq_data_entry cq_entry;

  return
    ofi_poll_cq_with_timeout (gctx,
                              fabric_ctx->qP->scq,
                              &cq_entry,
                              timeout_ms);
}


gaspi_return_t
pgaspi_dev_passive_receive (gaspi_context_t * const gctx,
                            const gaspi_segment_id_t segment_id_local,
                            const gaspi_offset_t offset_local,
                            gaspi_rank_t * const rem_rank,
                            const gaspi_size_t size,
                            const gaspi_timeout_t timeout_ms)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];

  void* const local_desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], gctx->rank);

  int ret;

  /* For a passive receive, as we don't know the sender, we need to
   * post of both fabrics (if in use) */

  //post remote recv
  struct ofi_fabric* fabric_ctx = ofi_ctx->fabric_ctx[0];

  if (fabric_ctx)
  {
    do
    {
      ret = fi_recv (fabric_ctx->qP->ep,
                     (void*)local_seg.data.addr + offset_local,
                     size,
                     local_desc,
                     FI_ADDR_UNSPEC,
                     NULL);
      if (ret && ret != -FI_EAGAIN)
      {
        GASPI_DEBUG_PRINT_ERROR
          ("Error initiating passive receive (%d: %s)\n",
           ret, fi_strerror (-ret));

        return GASPI_ERR_DEVICE;
      }
    } while (ret);
  }

  //post local recv
  fabric_ctx = ofi_ctx->fabric_ctx[1];

  if (fabric_ctx)
  {
    do
    {
      ret = fi_recv (fabric_ctx->qP->ep,
                     (void*)local_seg.data.addr + offset_local,
                     size,
                     local_desc,
                     FI_ADDR_UNSPEC,
                     NULL);
      if (ret && ret != -FI_EAGAIN)
      {
        GASPI_DEBUG_PRINT_ERROR
          ("Error initiating passive receive (%d: %s)\n",
           ret, fi_strerror (-ret));

        return GASPI_ERR_DEVICE;
      }
    } while (ret);
  }

  // Wait for a message to be received
  struct fi_cq_data_entry cq_entry;
  gaspi_return_t gret;
  if (NULL != ofi_ctx->fabric_ctx[1])
  {
     gret = ofi_poll_cqs_with_timeout (gctx,
                                       ofi_ctx->fabric_ctx[0]->qP->rcq,
                                       ofi_ctx->fabric_ctx[1]->qP->rcq,
                                       &cq_entry,
                                       timeout_ms);

    if (gret == GASPI_SUCCESS)
    {
      if (cq_entry.flags & FI_REMOTE_CQ_DATA)
      {
        uint64_t sender_data = cq_entry.data;
        *rem_rank = sender_data;
      }
      else
      {
        GASPI_DEBUG_PRINT_ERROR ("Unexpected error in passive receive\n");
        return GASPI_ERR_DEVICE;
      }
    }
  }

  else
  {
    gret = ofi_poll_cq_with_timeout (gctx,
                                     ofi_ctx->fabric_ctx[0]->qP->rcq,
                                     &cq_entry,
                                     timeout_ms);

    if (gret == GASPI_SUCCESS)
    {
      if (cq_entry.flags & FI_REMOTE_CQ_DATA)
      {
        uint64_t sender_data = cq_entry.data;
        *rem_rank = sender_data;
      }
      else
      {
        GASPI_DEBUG_PRINT_ERROR ("Unexpected error in passive receive\n");
        return GASPI_ERR_DEVICE;
      }
    }

  }

  return gret;
}
