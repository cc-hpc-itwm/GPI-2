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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <rdma/fi_atomic.h>

#include "GASPI_Ext.h"
#include "GPI2.h"
#include "GPI2_OFI.h"
#include "GPI2_SEG.h"
#include "GPI2_Utility.h"

static inline uint64_t
pgaspi_ofi_remote_addr (struct ofi_fabric* fabric_ctx,
                        const gaspi_rc_mseg_t* remote_seg,
                        gaspi_offset_t offset)
{
  if (fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
    return (uint64_t)(remote_seg->data.addr + offset);

  return (uint64_t) offset;
}

int
pgaspi_ofi_cq_readerr (struct fid_cq *cq, void** op_ctx)
{
  struct fi_cq_err_entry cq_err;
  memset (&cq_err, 0, sizeof (cq_err));

  int ret = fi_cq_readerr (cq, &cq_err, 0);
  if (ret < 0)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("fi_cq_readerr (%d): %s", ret, fi_strerror (ret));

    return ret;
  }

  GASPI_DEBUG_PRINT_ERROR
    ("ofi_cq_readerr %d (%s), provider error: %d (%s)",
     cq_err.err,
     fi_strerror (cq_err.err),
     cq_err.prov_errno,
     fi_cq_strerror (cq, cq_err.prov_errno, cq_err.err_data, NULL, 0));

  if (op_ctx != NULL)
  {
    *op_ctx = cq_err.op_context;
  }

  return 0;
}

/* A read of a read_notify ended: the last notifies, unless one failed. */
static void
pgaspi_ofi_read_notify_complete (gaspi_ofi_ctx* const ofi_ctx,
                                 struct ofi_read_notify* const rn,
                                 const int failed)
{
  const gaspi_queue_id_t queue = rn->queue;

  if (failed)
  {
    __atomic_store_n (&rn->failed, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch (&ofi_ctx->read_notify_failed[queue], 1,
                        __ATOMIC_RELAXED);
  }

  if (0 == __atomic_sub_fetch (&rn->reads, 1, __ATOMIC_ACQ_REL))
  {
    /* The data has arrived: a completion is never reordered before it. */
    if (!__atomic_load_n (&rn->failed, __ATOMIC_RELAXED))
    {
      __atomic_store_n (rn->slot, 1, __ATOMIC_RELEASE);
    }

    __atomic_store_n (&rn->busy, 0, __ATOMIC_RELEASE);
  }

  /* Last: a wait that sees the read done sees its notification as well. */
  __atomic_add_fetch (&ofi_ctx->read_notify_done[queue], 1, __ATOMIC_RELEASE);
  __atomic_sub_fetch (&ofi_ctx->read_notify_inflight, 1, __ATOMIC_RELAXED);
}

int
pgaspi_ofi_reap_read_notify (struct ofi_fabric* fabric_ctx)
{
  struct ofi_queue* const q =
    __atomic_load_n (&fabric_ctx->qReadNotify, __ATOMIC_ACQUIRE);

  if (NULL == q
      || lock_gaspi_tout (&fabric_ctx->read_notify_cq_lock, GASPI_TEST))
  {
    return 0;
  }

  gaspi_context_t* const gctx = &glb_gaspi_ctx;
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;

  struct fi_cq_data_entry comp[64];
  int reaped = 0;

  const ssize_t ret = fi_cq_read (q->scq, comp, 64);
  if (ret > 0)
  {
    for (ssize_t i = 0; i < ret; i++)
    {
      pgaspi_ofi_read_notify_complete (ofi_ctx, comp[i].op_context, 0);
    }
    reaped = ret;
  }
  else if (ret == -FI_EAVAIL)
  {
    void* op_ctx = NULL;
    if (0 == pgaspi_ofi_cq_readerr (q->scq, &op_ctx) && op_ctx != NULL)
    {
      struct ofi_read_notify* const rn = op_ctx;

      GASPI_DEBUG_PRINT_ERROR
        ("read_notify from rank %u on queue %u failed", rn->rank, rn->queue);
      gctx->state_vec[rn->queue][rn->rank] = GASPI_STATE_CORRUPT;

      pgaspi_ofi_read_notify_complete (ofi_ctx, rn, 1);
      reaped = 1;
    }
  }

  unlock_gaspi (&fabric_ctx->read_notify_cq_lock);

  return reaped;
}

/* A free read_notify record: there is one if the queue has room. */
static struct ofi_read_notify*
pgaspi_ofi_read_notify_record (gaspi_context_t* const gctx,
                               const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;
  const gaspi_uint n = gctx->config->queue_size_max;

  if (NULL == ofi_ctx->read_notify_records[queue])
  {
    ofi_ctx->read_notify_records[queue] =
      calloc (n, sizeof (struct ofi_read_notify));
    if (NULL == ofi_ctx->read_notify_records[queue])
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
      return NULL;
    }
  }

  struct ofi_read_notify* const records = ofi_ctx->read_notify_records[queue];

  for (gaspi_uint k = 0; k < n; k++)
  {
    const gaspi_uint i = (ofi_ctx->read_notify_next[queue] + k) % n;

    if (!__atomic_load_n (&records[i].busy, __ATOMIC_ACQUIRE))
    {
      records[i].busy = 1;
      ofi_ctx->read_notify_next[queue] = (i + 1) % n;
      return &records[i];
    }
  }

  return NULL;
}

/* Waits for and dequeues the queue's read_notify reads. */
static gaspi_return_t
pgaspi_ofi_wait_read_notify (gaspi_context_t* const gctx,
                             const gaspi_queue_id_t queue,
                             const gaspi_timeout_t timeout_ms)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;
  const int posted = ofi_ctx->read_notify_posted[queue];

  if (0 == posted)
  {
    return GASPI_SUCCESS;
  }

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  pgaspi_ofi_wait_begin();
  while (__atomic_load_n (&ofi_ctx->read_notify_done[queue], __ATOMIC_ACQUIRE)
         < posted)
  {
    for (int f = 0; f < GPI2_OFI_MAX_FABRICS; f++)
    {
      if (ofi_ctx->fabric_ctx[f])
      {
        pgaspi_ofi_reap_read_notify (ofi_ctx->fabric_ctx[f]);
      }
    }

    const float ms = (float) (gaspi_get_cycles() - s0) * gctx->cycles_to_msecs;
    if (ms > timeout_ms)
    {
      pgaspi_ofi_wait_end();
      return GASPI_TIMEOUT;
    }
  }
  pgaspi_ofi_wait_end();

  /* No read is posted meanwhile: the caller holds the queue's lock. */
  __atomic_sub_fetch (&ofi_ctx->read_notify_done[queue], posted,
                      __ATOMIC_RELAXED);
  ofi_ctx->read_notify_posted[queue] = 0;
  gctx->ne_count_c[queue] -= posted;

  if (__atomic_exchange_n (&ofi_ctx->read_notify_failed[queue], 0,
                           __ATOMIC_RELAXED))
  {
    return GASPI_ERROR;
  }

  return GASPI_SUCCESS;
}

static unsigned long
pgaspi_ofi_connect_timeout_ms (void)
{
  static unsigned long timeout_ms = 0;

  if (0 == timeout_ms)
  {
    const char* env = getenv ("GASPI_OFI_CONNECT_TIMEOUT_MS");
    char* end = NULL;
    const unsigned long val = env != NULL ? strtoul (env, &end, 10) : 0;

    timeout_ms = (env != NULL && *env != '\0' && *end == '\0' && val > 0)
      ? val : 1000;
  }

  return timeout_ms;
}

int
pgaspi_ofi_peer_dead (gaspi_context_t* const gctx,
                      const gaspi_rank_t rank,
                      gaspi_cycles_t* const since)
{
  /* Found dead before: no need to wait again. */
  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    return 1;
  }

  const gaspi_cycles_t now = gaspi_get_cycles();

  if (0 == *since)
  {
    *since = now;
    return 0;
  }

  const unsigned long timeout_ms = pgaspi_ofi_connect_timeout_ms();

  if ((float) (now - *since) * gctx->cycles_to_msecs < timeout_ms)
  {
    return 0;
  }

  *since = now;

  /* Marks the peer in the state vector if it is dead. */
  return GASPI_ERROR == gaspi_proc_ping (rank, timeout_ms);
}

void
pgaspi_ofi_post_failed (gaspi_context_t* const gctx,
                        const gaspi_queue_id_t queue,
                        const gaspi_rank_t rank)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  GASPI_DEBUG_PRINT_ERROR
    ("Rank %u unreachable: operation on queue %u fails", rank, queue);

  gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
  ofi_ctx->failed_entries[queue]++;
  gctx->ne_count_c[queue]++;
}

/* Ranks the queue wrote without completion (NULL: complete as usual). */
static struct ofi_unsig_targets*
pgaspi_ofi_unsig_targets (gaspi_context_t* const gctx,
                          const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  if (ofi_ctx->unsig_targets[queue] == NULL)
  {
    struct ofi_unsig_targets* t = calloc (1, sizeof (*t));
    if (t != NULL)
    {
      t->ranks = calloc (gctx->tnc, sizeof (gaspi_rank_t));
      t->dirty = calloc (gctx->tnc, sizeof (unsigned char));
      if (t->ranks == NULL || t->dirty == NULL)
      {
        free (t->ranks);
        free (t->dirty);
        free (t);
        t = NULL;
      }
    }
    ofi_ctx->unsig_targets[queue] = t;
  }

  return ofi_ctx->unsig_targets[queue];
}

/* Small writes go without completion (one each costs a third on cxi). A wait
   completes them with a fenced zero-length write per rank written, which
   completes after them, their errors first (rxm ignores FI_FENCE but
   completes in order). It targets the rank's notification region: the
   segment may be deleted already. Returns 0 or a post's error. */
static int
pgaspi_ofi_flush_unsignaled (gaspi_context_t* const gctx,
                             const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;
  struct ofi_unsig_targets* const t = ofi_ctx->unsig_targets[queue];

  if (t == NULL || t->num == 0)
  {
    return 0;
  }

  int err = 0;

  for (int i = 0; i < t->num; i++)
  {
    const gaspi_rank_t rank = t->ranks[i];
    t->dirty[rank] = 0;

    if (err)
    {
      continue;
    }

    if (pgaspi_ofi_peer_known_dead (gctx, rank))
    {
      pgaspi_ofi_post_failed (gctx, queue, rank);
      continue;
    }

    struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];
    const struct ofi_addr_info* const peer = &fabric_ctx->remote_info[rank];
    const gaspi_rc_mseg_t local_seg = gctx->nsrc;

    const uint64_t remote_addr =
      fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
      peer->flush_addr : 0;

    const uint64_t key = peer->flush_key;

    void* desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[1], rank);

    struct iovec iov = { local_seg.notif_spc.buf, 0 };
    struct fi_rma_iov rma_iov = { remote_addr, 0, key };
    struct fi_msg_rma rma_msg =
      { .msg_iov = &iov, .desc = &desc, .iov_count = 1,
        .addr = pgaspi_ofi_rma_target (fabric_ctx,
                                       fabric_ctx->io_fi_addr[queue][rank],
                                       rank),
        .rma_iov = &rma_iov, .rma_iov_count = 1,
        .context = PGASPI_OFI_RANK_CTX (rank), .data = 0 };

    struct fid_ep* ep = fabric_ctx->qC[queue]->ep;
    gaspi_cycles_t since = 0;
    ssize_t ret;

    do
    {
      ret = fi_writemsg (ep, &rma_msg, FI_FENCE | FI_COMPLETION | FI_INJECT);

      if (ret && ret != -FI_EAGAIN)
      {
        GASPI_DEBUG_PRINT_ERROR
          ("Error initiating the flush of rank %u (%ld: %s).",
           rank, ret, fi_strerror (-ret));
        err = ret;
        break;
      }

      if (ret)
      {
        if (pgaspi_ofi_peer_dead (gctx, rank, &since))
        {
          pgaspi_ofi_post_failed (gctx, queue, rank);
          break;
        }

        pgaspi_ofi_progress_queue (fabric_ctx->qC[queue]);
      }
    } while (ret);

    if (ret == 0)
    {
      gctx->ne_count_c[queue]++;
    }
  }

  t->num = 0;

  return err;
}

/* All completions read: writes without completion are complete too. */
static void
pgaspi_ofi_unsig_done (gaspi_context_t* const gctx,
                       const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  gctx->ne_count_c[queue] -= ofi_ctx->unsig_count[queue];
  ofi_ctx->unsig_count[queue] = 0;
}

static gaspi_return_t
pgaspi_dev_ofi_poll (gaspi_context_t * const gctx,
                     gaspi_queue_id_t queue,
                     int count,
                     gaspi_timeout_t timeout_ms,
                     int handle_error,
                     int* polled)
{
#define POLL_BATCH_SIZE 64
  struct fi_cq_data_entry comp[POLL_BATCH_SIZE];

  struct ofi_fabric* fabric_ctx = NULL;

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  int ret;
  int to_poll = count;
  int unsig_failed = 0;

  do
  {
    for (int fabric = 0; fabric < GPI2_OFI_MAX_FABRICS; fabric++)
    {
      fabric_ctx = ofi_ctx->fabric_ctx[fabric];

      if (fabric_ctx && fabric_ctx->qC[queue])
      {
        int batch = to_poll < POLL_BATCH_SIZE ? to_poll : POLL_BATCH_SIZE;
        ret = fi_cq_read (fabric_ctx->qC[queue]->scq, comp, batch);

        if (ret < 0 && ret != -FI_EAGAIN)
        {
          if (handle_error)
          {
            void* op_ctx = NULL;

            /* The failed operation is an entry of the queue too. */
            if (ret == -FI_EAVAIL &&
                0 == pgaspi_ofi_cq_readerr (fabric_ctx->qC[queue]->scq,
                                            &op_ctx))
            {
              const uintptr_t ctx = (uintptr_t) op_ctx;

              /* The context is the target rank (PGASPI_OFI_RANK_CTX). */
              if (op_ctx != NULL)
              {
                const gaspi_rank_t rank = (ctx & ~PGASPI_OFI_CTX_UNSIG) - 1;

                GASPI_DEBUG_PRINT_ERROR
                  ("Operation to rank %u on queue %u failed%s", rank, queue,
                   ctx & PGASPI_OFI_CTX_UNSIG ? " (without completion)" : "");
                gctx->state_vec[queue][rank] = GASPI_STATE_CORRUPT;
              }

              /* A write without completion is no entry: read the rest, then
                 fail. With cxi its error has no context: the flush of its
                 rank names it. */
              if ((ctx & PGASPI_OFI_CTX_UNSIG)
                  || (op_ctx == NULL && pgaspi_ofi_unsignaled_writes()))
              {
                unsig_failed = 1;
                continue;
              }

              to_poll--;
            }
            else
            {
              GASPI_DEBUG_PRINT_ERROR
                (" fi_cq_read error (%d): %s.\n",
                 ret,
                 fi_strerror (ret));
            }

            *polled = count - to_poll;
            return GASPI_ERROR;
          }
        }

        if (ret > 0)
        {
          to_poll -= ret;
        }
      }
    }

    if (to_poll)
    {
      const gaspi_cycles_t s1 = gaspi_get_cycles();
      const gaspi_cycles_t tdelta = s1 - s0;

      const float ms = (float) tdelta * gctx->cycles_to_msecs;

      if (ms > timeout_ms)
      {
        *polled = count - to_poll;
        return GASPI_TIMEOUT;
      }
    }
  } while (to_poll > 0);

  *polled = count - to_poll;

  return unsig_failed ? GASPI_ERROR : GASPI_SUCCESS;
}

/* Removes the entries of operations to dead peers; returns their number. */
static int
pgaspi_ofi_take_failed (gaspi_context_t * const gctx,
                        const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  const int failed = ofi_ctx->failed_entries[queue];

  ofi_ctx->failed_entries[queue] = 0;
  gctx->ne_count_c[queue] -= failed;

  return failed;
}

gaspi_return_t
pgaspi_dev_purge (gaspi_context_t * const gctx,
                  const gaspi_queue_id_t queue,
                  const gaspi_timeout_t timeout_ms)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  if (pgaspi_ofi_flush_unsignaled (gctx, queue))
  {
    return GASPI_ERR_DEVICE;
  }

  pgaspi_ofi_take_failed (gctx, queue);

  /* Completions to read: none for unsignaled writes or read_notify reads. */
  int count = gctx->ne_count_c[queue] - ofi_ctx->unsig_count[queue]
    - ofi_ctx->read_notify_posted[queue];

  int polled = 0;
  gaspi_return_t ret = GASPI_SUCCESS;

  if (count)
  {
    pgaspi_ofi_wait_begin();
    ret = pgaspi_dev_ofi_poll (gctx, queue, count, timeout_ms, 0, &polled);
    pgaspi_ofi_wait_end();
  }

  gctx->ne_count_c[queue] -= polled;

  if (polled == count)
  {
    pgaspi_ofi_unsig_done (gctx, queue);
  }

  /* Their errors are dropped too. */
  if (ret == GASPI_SUCCESS
      && pgaspi_ofi_wait_read_notify (gctx, queue, timeout_ms) == GASPI_TIMEOUT)
  {
    ret = GASPI_TIMEOUT;
  }

  return ret;
}

gaspi_return_t
pgaspi_dev_wait (gaspi_context_t * const gctx,
                 const gaspi_queue_id_t queue,
                 const gaspi_timeout_t timeout_ms)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  /* A flush that cannot be posted is no failed peer. */
  if (pgaspi_ofi_flush_unsignaled (gctx, queue))
  {
    return GASPI_ERR_DEVICE;
  }

  /* Like an error completion: the next wait drains the rest. */
  if (pgaspi_ofi_take_failed (gctx, queue) > 0)
  {
    return GASPI_ERROR;
  }

  /* Completions to read: none for unsignaled writes or read_notify reads. */
  int q_count = gctx->ne_count_c[queue] - ofi_ctx->unsig_count[queue]
    - ofi_ctx->read_notify_posted[queue];

  int polled = 0;
  gaspi_return_t ret = GASPI_SUCCESS;

  if (q_count)
  {
    pgaspi_ofi_wait_begin();
    ret = pgaspi_dev_ofi_poll (gctx, queue, q_count, timeout_ms, 1, &polled);
    pgaspi_ofi_wait_end();
  }

  gctx->ne_count_c[queue] -= polled;

  /* All completions read, the flushes' among them. */
  if (polled == q_count)
  {
    pgaspi_ofi_unsig_done (gctx, queue);
  }

  if (ret == GASPI_SUCCESS)
  {
    ret = pgaspi_ofi_wait_read_notify (gctx, queue, timeout_ms);
  }

  return ret;
}

gaspi_return_t
pgaspi_dev_write  (gaspi_context_t * const gctx,
                   const gaspi_segment_id_t segment_id_local,
                   const gaspi_offset_t offset_local,
                   const gaspi_rank_t rank,
                   const gaspi_segment_id_t segment_id_remote,
                   const gaspi_offset_t offset_remote,
                   const gaspi_size_t size,
                   const gaspi_queue_id_t queue)
{
  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    pgaspi_ofi_post_failed (gctx, queue, rank);
    return GASPI_SUCCESS;
  }

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];
  const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote][rank];

  const uint64_t remote_addr =
    pgaspi_ofi_remote_addr (fabric_ctx, &remote_seg, offset_remote);

  void* const local_addr = (void*) local_seg.data.addr + offset_local;

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;
  const fi_addr_t dest =
    pgaspi_ofi_rma_target (fabric_ctx,
                           fabric_ctx->io_fi_addr[queue][rank],
                           rank);

  void* desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], rank);
  // TODO: would be good to avoid this
  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[0].local :
    remote_seg.rkey[0].remote;

  int ret;

  /* Not fi_inject_write: without a completion, an error would lose the rank. */
  const size_t max_inject_size = fabric_ctx->info->tx_attr->inject_size;

  /* Without completion: the next wait flushes it. */
  const int unsig = size <= max_inject_size
    && pgaspi_ofi_unsignaled_writes()
    && NULL != pgaspi_ofi_unsig_targets (gctx, queue);

  struct iovec iov = { local_addr, size };
  struct fi_rma_iov rma_iov = { remote_addr, size, key };
  struct fi_msg_rma rma_msg =
    { .msg_iov = &iov, .desc = &desc, .iov_count = 1, .addr = dest,
      .rma_iov = &rma_iov, .rma_iov_count = 1,
      .context =
        unsig ? PGASPI_OFI_UNSIG_CTX (rank) : PGASPI_OFI_RANK_CTX (rank),
      .data = 0 };

  gaspi_cycles_t since = 0;
  do
  {
    if (size <= max_inject_size)
    {
      ret = fi_writemsg (ep, &rma_msg,
                         unsig ? FI_INJECT : FI_INJECT | FI_COMPLETION);
    }
    else
    {
      ret = fi_write (ep, local_addr, size, desc, dest, remote_addr, key,
                     PGASPI_OFI_RANK_CTX (rank));
    }
    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a write operation (%d: %s).",
         ret, fi_strerror (-ret));

      return ret;
    }

    if (ret)
    {
      if (pgaspi_ofi_peer_dead (gctx, rank, &since))
      {
        pgaspi_ofi_post_failed (gctx, queue, rank);
        return GASPI_SUCCESS;
      }

      pgaspi_ofi_progress_queue (fabric_ctx->qC[queue]);
    }
  } while (ret);

  gctx->ne_count_c[queue]++;

  if (unsig)
  {
    struct ofi_unsig_targets* const t = ofi_ctx->unsig_targets[queue];

    ofi_ctx->unsig_count[queue]++;
    if (!t->dirty[rank])
    {
      t->dirty[rank] = 1;
      t->ranks[t->num++] = rank;
    }
  }

  return GASPI_SUCCESS;
}


gaspi_return_t
pgaspi_dev_read (gaspi_context_t * const gctx,
                 const gaspi_segment_id_t segment_id_local,
                 const gaspi_offset_t offset_local,
                 const gaspi_rank_t rank,
                 const gaspi_segment_id_t segment_id_remote,
                 const gaspi_offset_t offset_remote,
                 const gaspi_size_t size,
                 const gaspi_queue_id_t queue)
{
  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    pgaspi_ofi_post_failed (gctx, queue, rank);
    return GASPI_SUCCESS;
  }

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];
  const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote][rank];

  const uint64_t remote_addr =
    pgaspi_ofi_remote_addr (fabric_ctx, &remote_seg, offset_remote);

  void* local_addr = (void*) local_seg.data.addr + offset_local;

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;
  const fi_addr_t dest =
    pgaspi_ofi_rma_target (fabric_ctx,
                           fabric_ctx->io_fi_addr[queue][rank],
                           rank);

  void* desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], rank);

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[0].local :
    remote_seg.rkey[0].remote;

  int ret;
  gaspi_cycles_t since = 0;
  do
  {
    ret = fi_read (ep, local_addr, size, desc, dest, remote_addr, key,
                  PGASPI_OFI_RANK_CTX (rank));

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a read (%d: %s).", ret, fi_strerror (-ret));
      return ret;
    }

    if (ret)
    {
      if (pgaspi_ofi_peer_dead (gctx, rank, &since))
      {
        pgaspi_ofi_post_failed (gctx, queue, rank);
        return GASPI_SUCCESS;
      }

      pgaspi_ofi_progress_queue (fabric_ctx->qC[queue]);
    }
  } while (ret);

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_write_list (gaspi_context_t * const gctx,
                       const gaspi_number_t num,
                       gaspi_segment_id_t * const segment_id_local,
                       gaspi_offset_t * const offset_local,
                       const gaspi_rank_t rank,
                       gaspi_segment_id_t * const segment_id_remote,
                       gaspi_offset_t * const offset_remote,
                       gaspi_size_t * const size,
                       const gaspi_queue_id_t queue)
{
  gaspi_return_t ret = GASPI_ERROR;

  /* All or nothing, as with Infiniband: each write is a queue entry. */
  if (gctx->ne_count_c[queue] + num > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  for (gaspi_number_t i = 0; i < num; i++)
  {
    if (size[i] == 0)
    {
      continue;
    }

    ret = pgaspi_dev_write (gctx,
                            segment_id_local[i], offset_local[i],
                            rank,
                            segment_id_remote[i],
                            offset_remote[i],
                            size[i],
                            queue);
    if (ret != GASPI_SUCCESS)
      {
        return ret;
      }
  }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read_list (gaspi_context_t * const gctx,
                      const gaspi_number_t num,
                      gaspi_segment_id_t * const segment_id_local,
                      gaspi_offset_t * const offset_local,
                      const gaspi_rank_t rank,
                      gaspi_segment_id_t * const segment_id_remote,
                      gaspi_offset_t * const offset_remote,
                      gaspi_size_t * const size,
                      const gaspi_queue_id_t queue)
{
  gaspi_return_t ret = GASPI_ERROR;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    if (size[i] == 0)
    {
      continue;
    }

    ret = pgaspi_dev_read (gctx,
                           segment_id_local[i], offset_local[i],
                           rank,
                           segment_id_remote[i], offset_remote[i],
                           size[i],
                           queue);
    if (ret != GASPI_SUCCESS)
    {
      return ret;
    }
  }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_notify (gaspi_context_t * const gctx,
                   const gaspi_segment_id_t segment_id_remote,
                   const gaspi_rank_t rank,
                   const gaspi_notification_id_t notification_id,
                   const gaspi_notification_t notification_value,
                   const gaspi_queue_id_t queue)
{
  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    pgaspi_ofi_post_failed (gctx, queue, rank);
    return GASPI_SUCCESS;
  }

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  *((gaspi_notification_slot_t *) (gctx->nsrc.notif_spc.buf +
                                   notification_id * sizeof (gaspi_notification_slot_t))) =
    notification_value;

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote][rank];
  const gaspi_rc_mseg_t local_seg = gctx->nsrc;


  uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t) (gctx->rrmd[segment_id_remote][rank].notif_spc.addr +
             notification_id * sizeof (gaspi_notification_slot_t)) :
    (uint64_t) (notification_id * sizeof (gaspi_notification_slot_t));

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[1].local :
    remote_seg.rkey[1].remote;

  void* desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[1], rank);

  struct fi_rma_iov rma_iov =
    { remote_addr,
      sizeof (gaspi_notification_slot_t),
      key
    };

  void* const local_addr =
    (void*) (gctx->nsrc.notif_spc.buf +
             notification_id * sizeof (gaspi_notification_slot_t));

  struct iovec iov;
  iov.iov_base = local_addr;
  iov.iov_len  = sizeof (gaspi_notification_slot_t);

  struct fi_msg_rma rma_msg;

  rma_msg.msg_iov = &iov;
  rma_msg.desc = &desc;
  rma_msg.iov_count = 1;
  rma_msg.addr =
    pgaspi_ofi_rma_target (fabric_ctx,
                           fabric_ctx->io_fi_addr[queue][rank],
                           rank);

  rma_msg.rma_iov = &rma_iov;
  rma_msg.rma_iov_count = 1;
  rma_msg.context = PGASPI_OFI_RANK_CTX (rank);
  rma_msg.data = 0;

  struct fi_ioc ioc = { local_addr, 1 };
  struct fi_rma_ioc rma_ioc = { remote_addr, 1, key };
  struct fi_msg_atomic atomic_msg =
    { .msg_iov = &ioc, .desc = &desc, .iov_count = 1,
      .addr = rma_msg.addr, .rma_iov = &rma_ioc, .rma_iov_count = 1,
      .datatype = FI_UINT64, .op = FI_ATOMIC_WRITE,
      .context = PGASPI_OFI_RANK_CTX (rank), .data = 0 };

  const int atomic = pgaspi_ofi_notify_atomic (ofi_ctx, fabric_ctx);

  /* rxm reports no completion for an injected atomic (it copies at post). */
  const uint64_t atomic_inject =
    fabric_ctx == ofi_ctx->fabric_ctx[1] ? FI_INJECT : 0;

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;
  ssize_t ret;
  gaspi_cycles_t since = 0;
  do
  {
    ret = atomic
      ? fi_atomicmsg (ep, &atomic_msg,
                      atomic_inject | FI_FENCE | FI_COMPLETION)
      : fi_writemsg (ep, &rma_msg, FI_INJECT | FI_FENCE | FI_COMPLETION);

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a notify (%ld: %s).", ret, fi_strerror (-ret));

      return ret;
    }

    if (ret)
    {
      if (pgaspi_ofi_peer_dead (gctx, rank, &since))
      {
        pgaspi_ofi_post_failed (gctx, queue, rank);
        return GASPI_SUCCESS;
      }

      pgaspi_ofi_progress_queue (fabric_ctx->qC[queue]);
    }
  } while (ret);

  /* With FI_COMPLETION: the CQ may be selective. */
  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

//write_notify as write+notify2 (writemsg with FENCE)
gaspi_return_t
pgaspi_dev_write_notify_separate (gaspi_context_t * const gctx,
                                  const gaspi_segment_id_t segment_id_local,
                                  const gaspi_offset_t offset_local,
                                  const gaspi_rank_t rank,
                                  const gaspi_segment_id_t segment_id_remote,
                                  const gaspi_offset_t offset_remote,
                                  const gaspi_size_t size,
                                  const gaspi_notification_id_t notification_id,
                                  const gaspi_notification_t notification_value,
                                  const gaspi_queue_id_t queue)
{
  gaspi_return_t ret = pgaspi_dev_write (gctx, segment_id_local, offset_local,
                                         rank,
                                         segment_id_remote, offset_remote,
                                         size,
                                         queue);
  if (ret != GASPI_SUCCESS)
  {
    return ret;
  }

  return pgaspi_dev_notify (gctx, segment_id_remote, rank,
                            notification_id, notification_value, queue);

}

gaspi_return_t
pgaspi_dev_write_notify (gaspi_context_t * const gctx,
                         const gaspi_segment_id_t segment_id_local,
                         const gaspi_offset_t offset_local,
                         const gaspi_rank_t rank,
                         const gaspi_segment_id_t segment_id_remote,
                         const gaspi_offset_t offset_remote,
                         const gaspi_size_t size,
                         const gaspi_notification_id_t notification_id,
                         const gaspi_notification_t notification_value,
                         const gaspi_queue_id_t queue)

{
  /* Data, then the notification as an operation of its own (an atomic write
     where the provider copies, see pgaspi_ofi_notify_atomic): one
     fi_writemsg may place the notification first. */
  const int entries = 2;

  if (gctx->ne_count_c[queue] + entries > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    pgaspi_ofi_post_failed (gctx, queue, rank);
    return GASPI_SUCCESS;
  }

  return pgaspi_dev_write_notify_separate (gctx,
                                             segment_id_local,
                                             offset_local,
                                             rank,
                                             segment_id_remote,
                                             offset_remote,
                                             size,
                                             notification_id,
                                             notification_value,
                                             queue);
}

gaspi_return_t
pgaspi_dev_write_list_notify (gaspi_context_t * const gctx,
                              const gaspi_number_t num,
                              gaspi_segment_id_t * const segment_id_local,
                              gaspi_offset_t * const offset_local,
                              const gaspi_rank_t rank,
                              gaspi_segment_id_t * const segment_id_remote,
                              gaspi_offset_t * const offset_remote,
                              gaspi_size_t * const size,
                              const gaspi_segment_id_t segment_id_notification,
                              const gaspi_notification_id_t notification_id,
                              const gaspi_notification_t notification_value,
                              const gaspi_queue_id_t queue)
{
  /* Room for the writes and the notification. */
  if (gctx->ne_count_c[queue] + num + 1 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_return_t ret =
    pgaspi_dev_write_list (gctx,
                           num,
                           segment_id_local, offset_local,
                           rank,
                           segment_id_remote, offset_remote,
                           size,
                           queue);
  if (ret != GASPI_SUCCESS)
  {
    return ret;
  }

  return pgaspi_dev_notify (gctx, segment_id_notification, rank,
                            notification_id, notification_value, queue);
}


/* read_notify where the notification is set locally: post the reads on
   qReadNotify and return. Each read is a queue entry; the thread completing
   the last one sets the notification. */
static gaspi_return_t
pgaspi_ofi_read_notify_deferred (gaspi_context_t * const gctx,
                                 const gaspi_number_t num,
                                 gaspi_segment_id_t * const segment_id_local,
                                 gaspi_offset_t * const offset_local,
                                 const gaspi_rank_t rank,
                                 gaspi_segment_id_t * const segment_id_remote,
                                 gaspi_offset_t * const offset_remote,
                                 gaspi_size_t * const size,
                                 const gaspi_segment_id_t segment_id_notification,
                                 const gaspi_notification_id_t notification_id,
                                 const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;
  struct ofi_fabric* const fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  int reads = 0;
  for (gaspi_number_t i = 0; i < num; i++)
  {
    reads += size[i] > 0;
  }

  /* All or nothing, as write_list. */
  if (gctx->ne_count_c[queue] + reads > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_notification_slot_t* const slot = (gaspi_notification_slot_t*)
    (gctx->rrmd[segment_id_notification][gctx->rank].notif_spc.buf
     + notification_id * sizeof (gaspi_notification_slot_t));

  if (0 == reads)
  {
    __atomic_store_n (slot, 1, __ATOMIC_RELEASE);
    return GASPI_SUCCESS;
  }

  if (pgaspi_ofi_peer_known_dead (gctx, rank))
  {
    for (int r = 0; r < reads; r++)
    {
      pgaspi_ofi_post_failed (gctx, queue, rank);
    }
    return GASPI_SUCCESS;
  }

  struct ofi_read_notify* const rn = pgaspi_ofi_read_notify_record (gctx, queue);
  if (NULL == rn)
  {
    return GASPI_ERR_MEMALLOC;
  }

  rn->reads = reads;
  rn->failed = 0;
  rn->queue = queue;
  rn->rank = rank;
  rn->slot = slot;

  struct fid_ep* const ep = fabric_ctx->qReadNotify->ep;
  const fi_addr_t dest =
    pgaspi_ofi_rma_target (fabric_ctx,
                           fabric_ctx->io_fi_addr[queue][rank],
                           rank);

  int posted = 0;
  gaspi_return_t eret = GASPI_SUCCESS;

  for (gaspi_number_t i = 0; i < num && eret == GASPI_SUCCESS; i++)
  {
    if (size[i] == 0)
    {
      continue;
    }

    const gaspi_rc_mseg_t local_seg =
      gctx->rrmd[segment_id_local[i]][gctx->rank];
    const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote[i]][rank];

    const uint64_t remote_addr =
      pgaspi_ofi_remote_addr (fabric_ctx, &remote_seg, offset_remote[i]);
    void* const local_addr = (void*) local_seg.data.addr + offset_local[i];
    void* desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], rank);

    const uint64_t key =
      pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
      remote_seg.rkey[0].local :
      remote_seg.rkey[0].remote;

    /* Before posting: the completion may be read at once. */
    __atomic_add_fetch (&ofi_ctx->read_notify_inflight, 1, __ATOMIC_RELAXED);

    ssize_t ret;
    gaspi_cycles_t since = 0;
    do
    {
      ret = fi_read (ep, local_addr, size[i], desc, dest, remote_addr, key, rn);

      if (ret && ret != -FI_EAGAIN)
      {
        GASPI_DEBUG_PRINT_ERROR
          ("Error initiating a read (%ld: %s).", ret, fi_strerror (-ret));
        eret = GASPI_ERR_DEVICE;
        break;
      }

      if (ret)
      {
        if (pgaspi_ofi_peer_dead (gctx, rank, &since))
        {
          eret = GASPI_ERROR;
          break;
        }

        pgaspi_ofi_reap_read_notify (fabric_ctx);
      }
    } while (ret);

    if (ret)
    {
      __atomic_sub_fetch (&ofi_ctx->read_notify_inflight, 1, __ATOMIC_RELAXED);
      break;
    }

    posted++;
    gctx->ne_count_c[queue]++;
    ofi_ctx->read_notify_posted[queue]++;
  }

  if (posted < reads)
  {
    /* Reads not posted fail the read_notify; dead peers' are failed entries. */
    if (eret == GASPI_ERROR)
    {
      for (int r = posted; r < reads; r++)
      {
        pgaspi_ofi_post_failed (gctx, queue, rank);
      }
      eret = GASPI_SUCCESS;
    }

    __atomic_store_n (&rn->failed, 1, __ATOMIC_RELAXED);
    if (0 == __atomic_sub_fetch (&rn->reads, reads - posted, __ATOMIC_ACQ_REL))
    {
      __atomic_store_n (&rn->busy, 0, __ATOMIC_RELEASE);
    }
  }

  return eret;
}

static gaspi_return_t
pgaspi_dev_remote_notify (gaspi_context_t * const gctx,
                          const gaspi_segment_id_t segment_id_local,
                          const gaspi_segment_id_t segment_id_remote,
                          const gaspi_rank_t rank,
                          const gaspi_notification_id_t notification_id,
                          const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];
  const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote][rank];

  gaspi_number_t notifications_space_size = pgaspi_notifications_space_size();
  const size_t notifications_offset =
    notifications_space_size - sizeof (gaspi_notification_slot_t);

  uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t) (remote_seg.notif_spc.addr + notifications_offset) :
    (uint64_t) notifications_offset;

  void* local_addr =
    (void*)local_seg.notif_spc.addr +
    notification_id * sizeof (gaspi_notification_slot_t);

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;
  fi_addr_t dest =
    pgaspi_ofi_rma_target (fabric_ctx,
                           fabric_ctx->io_fi_addr[queue][rank],
                           rank);

  void* const desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[1], rank);

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[1].local :
    remote_seg.rkey[1].remote;

  int ret;
  gaspi_cycles_t since = 0;
  do
  {
    ret = fi_read (ep,
                   local_addr,
                   sizeof (gaspi_notification_slot_t),
                   desc,
                   dest,
                   remote_addr,
                   key,
                   PGASPI_OFI_RANK_CTX (rank));

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a read (%d: %s).", ret, fi_strerror (-ret));
      return ret;
    }

    if (ret)
    {
      if (pgaspi_ofi_peer_dead (gctx, rank, &since))
      {
        pgaspi_ofi_post_failed (gctx, queue, rank);
        return GASPI_SUCCESS;
      }

      pgaspi_ofi_progress_queue (fabric_ctx->qC[queue]);
    }
  } while (ret);

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read_notify (gaspi_context_t * const gctx,
                        const gaspi_segment_id_t segment_id_local,
                        const gaspi_offset_t offset_local,
                        const gaspi_rank_t rank,
                        const gaspi_segment_id_t segment_id_remote,
                        const gaspi_offset_t offset_remote,
                        const gaspi_size_t size,
                        const gaspi_notification_id_t notification_id,
                        const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;

  if (pgaspi_ofi_defers_read_notify (ofi_ctx, ofi_ctx->rank_fabric_map[rank]))
  {
    gaspi_segment_id_t seg_local = segment_id_local;
    gaspi_offset_t off_local = offset_local;
    gaspi_segment_id_t seg_remote = segment_id_remote;
    gaspi_offset_t off_remote = offset_remote;
    gaspi_size_t sz = size;

    return pgaspi_ofi_read_notify_deferred (gctx, 1,
                                            &seg_local, &off_local,
                                            rank,
                                            &seg_remote, &off_remote,
                                            &sz,
                                            segment_id_local,
                                            notification_id, queue);
  }

  gaspi_return_t ret = pgaspi_dev_read (gctx,
                                        segment_id_local, offset_local,
                                        rank,
                                        segment_id_remote, offset_remote,
                                        size,
                                        queue);
  if (ret != GASPI_SUCCESS)
  {
    return ret;
  }

  /* Read, then notification, without a wait: relies on the endpoint's order.
     Where there is none, the notification is set locally
     (pgaspi_ofi_read_notify_deferred). */

  return pgaspi_dev_remote_notify (gctx,
                                   segment_id_local, segment_id_remote,
                                   rank,
                                   notification_id, queue);
}

gaspi_return_t
pgaspi_dev_read_list_notify (gaspi_context_t * const gctx,
                             const gaspi_number_t num,
                             gaspi_segment_id_t * const segment_id_local,
                             gaspi_offset_t * const offset_local,
                             const gaspi_rank_t rank,
                             gaspi_segment_id_t * const segment_id_remote,
                             gaspi_offset_t * const offset_remote,
                             gaspi_size_t * const size,
                             const gaspi_segment_id_t segment_id_notification,
                             const gaspi_notification_id_t notification_id,
                             const gaspi_queue_id_t queue)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;

  if (pgaspi_ofi_defers_read_notify (ofi_ctx, ofi_ctx->rank_fabric_map[rank]))
  {
    return pgaspi_ofi_read_notify_deferred (gctx, num,
                                            segment_id_local, offset_local,
                                            rank,
                                            segment_id_remote, offset_remote,
                                            size,
                                            segment_id_notification,
                                            notification_id, queue);
  }

  gaspi_return_t ret =
    pgaspi_dev_read_list (gctx,
                          num,
                          segment_id_local, offset_local,
                          rank,
                          segment_id_remote, offset_remote,
                          size,
                          queue);
  if (ret != GASPI_SUCCESS)
  {
    return ret;
  }

  /* Read, then notification, without a wait: relies on the endpoint's order.
     Where there is none, the notification is set locally
     (pgaspi_ofi_read_notify_deferred). */

  return
    pgaspi_dev_remote_notify (gctx,
                              segment_id_notification,
                              segment_id_notification,
                              rank,
                              notification_id,
                              queue);
}

/* Drives shm, which processes some operations (atomic notifications) at the
   target, for a waiting thread; skips queues the progress thread holds. Then
   sets the notifications of completed read_notify. */
void
pgaspi_dev_progress_waiting (gaspi_context_t const * const gctx)
{
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;
  struct ofi_fabric* const local_fabric = ofi_ctx->fabric_ctx[1];

  if (local_fabric != NULL)
  {
    for (uint32_t q = 0; q < local_fabric->num_qC; q++)
    {
      if (local_fabric->qC[q] != NULL
          && !lock_gaspi_tout (&local_fabric->qCQ_lock[q], GASPI_TEST))
      {
        pgaspi_ofi_progress_queue (local_fabric->qC[q]);
        unlock_gaspi (&local_fabric->qCQ_lock[q]);
      }
    }

    pgaspi_ofi_progress_queue (local_fabric->qAtomic);
  }

  if (__atomic_load_n (&ofi_ctx->read_notify_inflight, __ATOMIC_RELAXED))
  {
    for (int f = 0; f < GPI2_OFI_MAX_FABRICS; f++)
    {
      if (ofi_ctx->fabric_ctx[f])
      {
        pgaspi_ofi_reap_read_notify (ofi_ctx->fabric_ctx[f]);
      }
    }
  }
}
