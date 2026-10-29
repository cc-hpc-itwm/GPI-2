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

#include "GPI2.h"
#include "GPI2_Mem.h"
#include "GPI2_OFI.h"
#include "GPI2_Utility.h"

//TODO: there is some repetition (polling and posting requests) that
//could be improved

/* An error completion fails the operation; anything else is a device error. */
static gaspi_return_t
ofi_passive_cq_error (struct fid_cq* const cq, const int ret)
{
  if (ret == -FI_EAVAIL && 0 == pgaspi_ofi_cq_readerr (cq, NULL))
  {
    return GASPI_ERROR;
  }

  GASPI_DEBUG_PRINT_ERROR
    ("Passive CQ read error (%d: %s)\n", ret, fi_strerror (-ret));

  return GASPI_ERR_DEVICE;
}

/* Another send's completion: that timed-out send is no longer pending. */
static void
ofi_passive_send_done (gaspi_context_t * const gctx,
                       gaspi_ofi_ctx* const ofi_ctx,
                       void* const op_ctx)
{
  for (gaspi_rank_t r = 0; r < gctx->tnc; r++)
  {
    if (ofi_ctx->passive_send_pending[r] == (uintptr_t) op_ctx)
    {
      ofi_ctx->passive_send_pending[r] = 0;
      return;
    }
  }
}

/* Waits for passive send ctx to rank; on a timeout it stays pending for the
   next send. A dead receiver (local check or ping) fails it: shm needs the
   receiver to copy large messages. */
static gaspi_return_t
ofi_passive_send_wait (gaspi_context_t * const gctx,
                       struct fid_cq* const cq,
                       void* const ctx,
                       const gaspi_rank_t rank,
                       const gaspi_timeout_t timeout_ms)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;
  const gaspi_cycles_t s0 = gaspi_get_cycles();
  gaspi_cycles_t since = 0;

  ofi_ctx->passive_send_pending[rank] = (uintptr_t) ctx;

  for (;;)
  {
    struct fi_cq_data_entry cq_entry;
    const int ret = fi_cq_read (cq, &cq_entry, 1);

    if (ret == 1)
    {
      if (cq_entry.op_context == ctx)
      {
        ofi_ctx->passive_send_pending[rank] = 0;
        return GASPI_SUCCESS;
      }
      ofi_passive_send_done (gctx, ofi_ctx, cq_entry.op_context);
      continue;
    }

    if (ret == -FI_EAVAIL)
    {
      void* op_ctx = NULL;
      if (0 != pgaspi_ofi_cq_readerr (cq, &op_ctx))
      {
        return GASPI_ERR_DEVICE;
      }
      if (op_ctx == ctx)
      {
        ofi_ctx->passive_send_pending[rank] = 0;
        return GASPI_ERROR;
      }
      ofi_passive_send_done (gctx, ofi_ctx, op_ctx);
      continue;
    }

    if (ret != -FI_EAGAIN)
    {
      return ofi_passive_cq_error (cq, ret);
    }

    /* Marked dead by the ping. */
    if (pgaspi_ofi_peer_dead (gctx, rank, &since))
    {
      ofi_ctx->passive_send_pending[rank] = 0;
      return GASPI_ERROR;
    }

    const float ms = (float) (gaspi_get_cycles() - s0) * gctx->cycles_to_msecs;
    if (ms > timeout_ms)
    {
      return GASPI_TIMEOUT;
    }
  }
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

  struct ofi_fabric* fabric_ctx = ofi_ctx->passive_on_remote ?
    ofi_ctx->fabric_ctx[0] : ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];

  void* const local_desc = ofi_ctx->passive_on_remote ?
    pgaspi_ofi_get_remote_mr_desc (local_seg.mr[0]) :
    pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], rank);

  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    ofi_ctx->passive_send_pending[rank] = 0;
    return GASPI_ERROR;
  }

  /* A posted send to rank that timed out: wait for it, as the IB device
     does. */
  void* ctx = (void*) ofi_ctx->passive_send_pending[rank];
  if (ctx != NULL)
  {
    goto waitL;
  }

  ctx = (void*) ++ofi_ctx->passive_send_seq;

  const gaspi_cycles_t s0 = gaspi_get_cycles();
  gaspi_cycles_t since = 0;
  int ret;

  do
  {
    ret = fi_senddata (fabric_ctx->qP->ep,
                       (void*)local_seg.data.addr + offset_local,
                       size,
                       local_desc,
                       gctx->rank,
                       fabric_ctx->passive_fi_addr[rank],
                       ctx);
    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating passive send (%d: %s)\n",
         ret, fi_strerror (-ret));

      return GASPI_ERR_DEVICE;
    }

    if (ret)
    {
      /* Marked dead by the ping. */
      if (pgaspi_ofi_peer_dead (gctx, rank, &since))
      {
        return GASPI_ERROR;
      }

      const float ms =
        (float) (gaspi_get_cycles() - s0) * gctx->cycles_to_msecs;
      if (ms > timeout_ms)
      {
        return GASPI_TIMEOUT;
      }

      pgaspi_ofi_progress_queue (fabric_ctx->qP);
    }
  } while (ret);

waitL:
  pgaspi_ofi_wait_begin();
  const gaspi_return_t gret =
    ofi_passive_send_wait (gctx, fabric_ctx->qP->scq, ctx, rank, timeout_ms);
  pgaspi_ofi_wait_end();

  return gret;
}


int
pgaspi_ofi_passive_init (gaspi_context_t * const gctx)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;
  gaspi_rc_mseg_t* const buf = &ofi_ctx->passive_recv_buf;

  if (ofi_ctx->passive_on_remote)
  {
    return 0;
  }

  /* Fabric 1's buffer, of the largest passive message. */
  const size_t size =
    (gctx->config->passive_transfer_size_max + 63) & ~((size_t) 63);

  if (pgaspi_alloc_page_aligned (&buf->data.ptr, size) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Memory allocation failed.");
    return -1;
  }

  buf->size = size;

  if (pgaspi_dev_register_mem (gctx, buf) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to register passive receive buffers.");
    free (buf->data.buf);
    buf->data.buf = NULL;
    return -1;
  }

  return 0;
}

void
pgaspi_ofi_passive_cleanup (gaspi_context_t * const gctx)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;
  gaspi_rc_mseg_t* const buf = &ofi_ctx->passive_recv_buf;

  if (NULL == buf->data.buf)
  {
    return;
  }

  if (pgaspi_dev_unregister_mem (gctx, buf) != 0)
  {
    GASPI_PRINT_WARNING ("Failed to unregister passive receive buffers.");
  }

  free (buf->data.buf);
  buf->data.buf = NULL;
}

/* Posts fabric 1's receive into its buffer, unless it is posted or holds a
   kept message. */
static gaspi_return_t
ofi_passive_post_recv (gaspi_ofi_ctx* const ofi_ctx)
{
  struct ofi_fabric* const fabric_ctx = ofi_ctx->fabric_ctx[1];

  if (NULL == fabric_ctx || ofi_ctx->passive_on_remote
      || ofi_ctx->passive_recv_posted || ofi_ctx->passive_recv_kept)
  {
    return GASPI_SUCCESS;
  }

  void* const desc =
    pgaspi_ofi_get_local_mr_desc (ofi_ctx->passive_recv_buf.mr[0]);
  int ret;

  do
  {
    ret = fi_recv (fabric_ctx->qP->ep,
                   ofi_ctx->passive_recv_buf.data.buf,
                   ofi_ctx->passive_recv_buf.size,
                   desc,
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

  ofi_ctx->passive_recv_posted = 1;

  return GASPI_SUCCESS;
}

/* Posts fabric 0's receive into the caller's segment. */
static gaspi_return_t
ofi_passive_post_direct (gaspi_ofi_ctx* const ofi_ctx,
                         void* const dest,
                         const gaspi_size_t size,
                         void* const mr)
{
  struct ofi_fabric* const fabric_ctx = ofi_ctx->fabric_ctx[0];
  int ret;

  do
  {
    ret = fi_recv (fabric_ctx->qP->ep,
                   dest,
                   size,
                   pgaspi_ofi_get_remote_mr_desc (mr),
                   FI_ADDR_UNSPEC,
                   &ofi_ctx->passive_recv_ctx);
    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating passive receive (%d: %s)\n",
         ret, fi_strerror (-ret));

      return GASPI_ERR_DEVICE;
    }
  } while (ret);

  return GASPI_SUCCESS;
}

#define OFI_PASSIVE_CANCEL_MS 1000

/* Cancels fabric 0's receive: 1 if its message arrived (in entry), 0 if it
   was cancelled, GASPI_ERROR on an error completion, GASPI_ERR_DEVICE if
   its end was not seen. */
static int
ofi_passive_cancel_direct (gaspi_context_t * const gctx,
                           gaspi_ofi_ctx* const ofi_ctx,
                           struct fi_cq_data_entry* const entry)
{
  struct ofi_fabric* const fabric_ctx = ofi_ctx->fabric_ctx[0];
  struct fid_cq* const cq = fabric_ctx->qP->rcq;

  const int cret =
    fi_cancel (&fabric_ctx->qP->ep->fid, &ofi_ctx->passive_recv_ctx);
  if (cret && cret != -FI_ENOENT)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to cancel passive receive (%d: %s)\n",
       cret, fi_strerror (-cret));
    return GASPI_ERR_DEVICE;
  }

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  for (;;)
  {
    const int ret = fi_cq_read (cq, entry, 1);

    if (ret == 1)
    {
      return 1;
    }

    if (ret == -FI_EAVAIL)
    {
      struct fi_cq_err_entry err;
      memset (&err, 0, sizeof (err));

      if (fi_cq_readerr (cq, &err, 0) < 0)
      {
        return GASPI_ERR_DEVICE;
      }

      return err.err == FI_ECANCELED ? 0 : GASPI_ERROR;
    }

    if (ret != -FI_EAGAIN)
    {
      return ofi_passive_cq_error (cq, ret);
    }

    const float ms = (float) (gaspi_get_cycles() - s0) * gctx->cycles_to_msecs;
    if (ms > OFI_PASSIVE_CANCEL_MS)
    {
      GASPI_DEBUG_PRINT_ERROR ("Cancelled passive receive did not end.");
      return GASPI_ERR_DEVICE;
    }
  }
}

/* A message of fabric 0, received in place. */
static gaspi_return_t
ofi_passive_direct_done (const struct fi_cq_data_entry* const entry,
                         gaspi_rank_t * const rem_rank)
{
  if (!(entry->flags & FI_REMOTE_CQ_DATA))
  {
    GASPI_DEBUG_PRINT_ERROR ("Unexpected error in passive receive\n");
    return GASPI_ERR_DEVICE;
  }

  *rem_rank = entry->data;

  return GASPI_SUCCESS;
}

/* A message of fabric 1, copied from its buffer; its receive posted again
   (a failure leaves it unposted, to be posted by the next call). */
static gaspi_return_t
ofi_passive_copy (gaspi_ofi_ctx* const ofi_ctx,
                  void* const dest,
                  const gaspi_size_t size,
                  const size_t len,
                  const gaspi_rank_t sender,
                  gaspi_rank_t * const rem_rank)
{
  /* Larger than the caller's buffer: dropped, as a truncated receive. */
  if (len > size)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Passive message of %lu bytes for a receive of %lu\n",
       (unsigned long) len, (unsigned long) size);
    ofi_passive_post_recv (ofi_ctx);
    return GASPI_ERROR;
  }

  memcpy (dest, ofi_ctx->passive_recv_buf.data.buf, len);

  *rem_rank = sender;

  ofi_passive_post_recv (ofi_ctx);

  return GASPI_SUCCESS;
}

/* The sender may be local or remote. Fabric 0 receives into the caller's
   segment and its receive never outlives the call; fabric 1 receives into
   its buffer. A fabric 1 message that came while fabric 0's arrived is kept
   for the next call. */
gaspi_return_t
pgaspi_dev_passive_receive (gaspi_context_t * const gctx,
                            const gaspi_segment_id_t segment_id_local,
                            const gaspi_offset_t offset_local,
                            gaspi_rank_t * const rem_rank,
                            const gaspi_size_t size,
                            const gaspi_timeout_t timeout_ms)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;
  const gaspi_rc_mseg_t* const seg =
    &gctx->rrmd[segment_id_local][gctx->rank];
  void* const dest = seg->data.buf + offset_local;
  struct ofi_fabric* const direct = ofi_ctx->fabric_ctx[0];
  struct ofi_fabric* const local =
    ofi_ctx->passive_on_remote ? NULL : ofi_ctx->fabric_ctx[1];

  if (ofi_ctx->passive_recv_kept)
  {
    ofi_ctx->passive_recv_kept = 0;
    return ofi_passive_copy (ofi_ctx, dest, size,
                             ofi_ctx->passive_recv_kept_len,
                             ofi_ctx->passive_recv_kept_rank, rem_rank);
  }

  gaspi_return_t gret = ofi_passive_post_recv (ofi_ctx);
  if (gret != GASPI_SUCCESS)
  {
    return gret;
  }

  if (direct != NULL)
  {
    gret = ofi_passive_post_direct (ofi_ctx, dest, size, seg->mr[0]);
    if (gret != GASPI_SUCCESS)
    {
      return gret;
    }
  }

  enum { GOT_DIRECT, GOT_LOCAL, LOCAL_FAILED, TIMED_OUT } got;
  struct fi_cq_data_entry entry, local_entry;
  const gaspi_cycles_t s0 = gaspi_get_cycles();

  /* Read every fabric before checking the timeout (GASPI_TEST). */
  pgaspi_ofi_wait_begin();
  for (;;)
  {
    if (direct != NULL)
    {
      const int ret = fi_cq_read (direct->qP->rcq, &entry, 1);

      if (ret == 1)
      {
        pgaspi_ofi_wait_end();
        return ofi_passive_direct_done (&entry, rem_rank);
      }

      if (ret != -FI_EAGAIN)
      {
        pgaspi_ofi_wait_end();
        return ofi_passive_cq_error (direct->qP->rcq, ret);
      }
    }

    if (local != NULL)
    {
      const int ret = fi_cq_read (local->qP->rcq, &local_entry, 1);

      if (ret == 1)
      {
        ofi_ctx->passive_recv_posted = 0;
        got = GOT_LOCAL;
        break;
      }

      if (ret != -FI_EAGAIN)
      {
        ofi_ctx->passive_recv_posted = 0;
        gret = ofi_passive_cq_error (local->qP->rcq, ret);
        got = LOCAL_FAILED;
        break;
      }
    }

    const float ms = (float) (gaspi_get_cycles() - s0) * gctx->cycles_to_msecs;
    if (ms > timeout_ms)
    {
      got = TIMED_OUT;
      break;
    }
  }

  const int local_ok =
    got == GOT_LOCAL && (local_entry.flags & FI_REMOTE_CQ_DATA);

  if (direct != NULL)
  {
    const int cret = ofi_passive_cancel_direct (gctx, ofi_ctx, &entry);

    if (cret != 0)
    {
      pgaspi_ofi_wait_end();

      if (local_ok)
      {
        ofi_ctx->passive_recv_kept = 1;
        ofi_ctx->passive_recv_kept_len = local_entry.len;
        ofi_ctx->passive_recv_kept_rank = local_entry.data;
      }
      else if (got == GOT_LOCAL)
      {
        ofi_passive_post_recv (ofi_ctx);
      }

      return cret == 1 ? ofi_passive_direct_done (&entry, rem_rank) : cret;
    }
  }
  pgaspi_ofi_wait_end();

  if (got == TIMED_OUT)
  {
    return GASPI_TIMEOUT;
  }

  if (got == LOCAL_FAILED)
  {
    return gret;
  }

  if (!local_ok)
  {
    GASPI_DEBUG_PRINT_ERROR ("Unexpected error in passive receive\n");
    ofi_passive_post_recv (ofi_ctx);
    return GASPI_ERR_DEVICE;
  }

  return ofi_passive_copy (ofi_ctx, dest, size, local_entry.len,
                           local_entry.data, rem_rank);
}
