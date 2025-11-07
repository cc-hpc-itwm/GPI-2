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
#include "GPI2_SEG.h"
#include "GPI2_Utility.h"

#define GPI2_OFI_MR_REMOTE_ADDR(fabric, segment, offset)        \
  fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?    \
  (uint64_t)(remote_seg.data.addr + offset_remote) :            \
  (uint64_t) offset_remote;

void
pgaspi_ofi_cq_readerr (struct fid_cq *cq)
{
  struct fi_cq_err_entry cq_err;
  memset (&cq_err, 0, sizeof (cq_err));

  int ret = fi_cq_readerr (cq, &cq_err, 0);
  if (ret < 0)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("fi_cq_readerr (%d): %s", ret, fi_strerror (ret));
  }
  else
  {
    GASPI_DEBUG_PRINT_ERROR
      ("ofi_cq_readerr %d (%s), provider error: %d (%s)",
       cq_err.err,
       fi_strerror (cq_err.err),
       cq_err.prov_errno,
       fi_cq_strerror (cq, cq_err.prov_errno, cq_err.err_data, NULL, 0));

    ret = -cq_err.err;
  }
}

static gaspi_return_t
pgaspi_dev_ofi_poll (gaspi_context_t * const gctx,
                     gaspi_queue_id_t queue,
                     int count,
                     gaspi_timeout_t timeout_ms,
                     int handle_error,
                     int* polled)
{
  struct fi_cq_data_entry* comp =
    malloc (count * sizeof (struct fi_cq_data_entry));
  if (NULL == comp)
  {
    return GASPI_ERR_MEMALLOC;
  }

  struct ofi_fabric* fabric_ctx = NULL;

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  int ret;
  int to_poll = count;

  do
  {
    for (int fabric = 0; fabric < GPI2_OFI_MAX_FABRICS; fabric++)
    {
      fabric_ctx = ofi_ctx->fabric_ctx[fabric];

      if (fabric_ctx && fabric_ctx->qC[queue])
      {
        ret = fi_cq_read (fabric_ctx->qC[queue]->scq, comp, to_poll);

        if (ret < 0 && ret != -FI_EAGAIN)
        {
          if (handle_error)
          {
            if (ret == -FI_EAVAIL)
            {
              pgaspi_ofi_cq_readerr (fabric_ctx->qC[queue]->scq);
            }
            else
            {
              GASPI_DEBUG_PRINT_ERROR
                (" fi_cq_read error (%d): %s.\n",
                 ret,
                 fi_strerror (ret));
            }

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

  free (comp);
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_purge (gaspi_context_t * const gctx,
                  const gaspi_queue_id_t queue,
                  const gaspi_timeout_t timeout_ms)
{
  int count = gctx->ne_count_c[queue];

  if (!count)
  {
    return GASPI_SUCCESS;
  }

  int polled = 0;

  gaspi_return_t ret =
    pgaspi_dev_ofi_poll (gctx, queue, count, timeout_ms, 0, &polled);

  gctx->ne_count_c[queue] -= polled;

  return ret;
}

gaspi_return_t
pgaspi_dev_wait (gaspi_context_t * const gctx,
                 const gaspi_queue_id_t queue,
                 const gaspi_timeout_t timeout_ms)
{
  int q_count = gctx->ne_count_c[queue];

  if (!q_count)
  {
    return GASPI_SUCCESS;
  }

  int polled = 0;

  gaspi_return_t ret =
    pgaspi_dev_ofi_poll (gctx, queue, q_count, timeout_ms, 1, &polled);

  gctx->ne_count_c[queue] -= polled;

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

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];
  const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote][rank];

  const uint64_t remote_addr =
    GPI2_OFI_MR_REMOTE_ADDR (fabric_ctx, remote_seg, offset_remote);

  void* const local_addr = (void*) local_seg.data.addr + offset_local;

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;
  const fi_addr_t dest = fabric_ctx->io_fi_addr[queue][rank];

  void* desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], rank);
  // TODO: would be good to avoid this
  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[0].local :
    remote_seg.rkey[0].remote;

  int ret;


  const size_t max_inject_size = fabric_ctx->info->tx_attr->inject_size;

  do
  {
    if (size <= max_inject_size)
    {
      ret = fi_inject_write (ep, local_addr, size, dest, remote_addr, key);
    }
    else
    {
      ret = fi_write (ep, local_addr, size, desc, dest, remote_addr, key, NULL);
    }
    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a write operation (%d: %s).",
         ret, fi_strerror (-ret));

      return ret;
    }
  } while (ret);

  if (size > max_inject_size)
  {
    gctx->ne_count_c[queue]++;
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

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];
  const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote][rank];

  const uint64_t remote_addr =
    GPI2_OFI_MR_REMOTE_ADDR (fabric_ctx, remote_seg, offset_remote);

  void* local_addr = (void*) local_seg.data.addr + offset_local;

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;
  const fi_addr_t dest = fabric_ctx->io_fi_addr[queue][rank];

  void* desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], rank);

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[0].local :
    remote_seg.rkey[0].remote;

  int ret;
  do
  {
    ret = fi_read (ep, local_addr, size, desc, dest, remote_addr, key, NULL);

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a read (%d: %s).", ret, fi_strerror (-ret));
      return ret;
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
  gaspi_number_t num_entries = num;

  gaspi_return_t ret = GASPI_ERROR;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    if (size[i] == 0)
    {
      num_entries--;
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
  gaspi_number_t num_entries = num;

  gaspi_return_t ret = GASPI_ERROR;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    if (size[i] == 0)
    {
      num_entries--;
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
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  *((gaspi_notification_t *) (gctx->nsrc.notif_spc.buf +
                              notification_id * sizeof (gaspi_notification_t))) = notification_value;

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote][rank];
  const gaspi_rc_mseg_t local_seg = gctx->nsrc;


  uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t) (gctx->rrmd[segment_id_remote][rank].notif_spc.addr +
             notification_id * sizeof (gaspi_notification_t)) :
    (uint64_t) (notification_id * sizeof (gaspi_notification_t));

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[1].local :
    remote_seg.rkey[1].remote;

  void* desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[1], rank);

  struct fi_rma_iov rma_iov =
    { remote_addr,
      sizeof (gaspi_notification_t),
      key
    };

  void* const local_addr =
    (void*) (gctx->nsrc.notif_spc.buf +
             notification_id * sizeof (gaspi_notification_t));

  struct iovec iov;
  iov.iov_base = local_addr;
  iov.iov_len  = sizeof(gaspi_notification_t);

  struct fi_msg_rma rma_msg;

  rma_msg.msg_iov = &iov;
  rma_msg.desc = &desc;
  rma_msg.iov_count = 1;
  rma_msg.addr = fabric_ctx->io_fi_addr[queue][rank];;

  rma_msg.rma_iov = &rma_iov;
  rma_msg.rma_iov_count = 1;
  rma_msg.context = NULL;
  rma_msg.data = 0;

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;
  ssize_t ret;
  do
  {
    ret = fi_writemsg (ep, &rma_msg, FI_INJECT | FI_FENCE);

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a notify (%ld: %s).", ret, fi_strerror (-ret));

      return ret;
    }
  } while (ret);

  return GASPI_SUCCESS;
}

//write_notify as write+notify2 (writemsg with FENCE)
gaspi_return_t
pgaspi_dev_write_notify_no_rma_iov (gaspi_context_t * const gctx,
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

//using writemsg with 2 elems in iovec and FENCE
//does not work with providers that do not support more 1 (count) in rma_iov
gaspi_return_t
pgaspi_dev_write_notify_with_rma_iov (gaspi_context_t * const gctx,
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
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  *((gaspi_notification_t *) (gctx->nsrc.notif_spc.buf +
                              notification_id * sizeof (gaspi_notification_t))) = notification_value;

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  const gaspi_rc_mseg_t remote_seg = gctx->rrmd[segment_id_remote][rank];
  const gaspi_rc_mseg_t local_seg = gctx->rrmd[segment_id_local][gctx->rank];

  const uint64_t remote_addr =
    GPI2_OFI_MR_REMOTE_ADDR (fabric_ctx, remote_seg, offset_remote);


  uint64_t remote_notf_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t) (gctx->rrmd[segment_id_remote][rank].notif_spc.addr +
             notification_id * sizeof (gaspi_notification_t)) :
    (uint64_t) (notification_id * sizeof (gaspi_notification_t));

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[0].local :
    remote_seg.rkey[0].remote;

  uint64_t key_notf =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[1].local :
    remote_seg.rkey[1].remote;

  void* desc[2];
  desc[0] = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[0], rank);
  desc[1] = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[1], rank);

  struct fi_rma_iov rma_iov[2];

  rma_iov[0].addr = remote_addr;
  rma_iov[0].len = size;
  rma_iov[0].key = key;

  rma_iov[1].addr = remote_notf_addr;
  rma_iov[1].len = sizeof (gaspi_notification_t);
  rma_iov[1].key = key_notf;

  struct iovec iov[2];

  //data
  void* const local_addr = (void*) local_seg.data.addr + offset_local;

  iov[0].iov_base = local_addr;
  iov[0].iov_len  = size;

  //notification
  void* const local_notf_addr =
    (void*) (gctx->nsrc.notif_spc.buf +
             notification_id * sizeof (gaspi_notification_t));

  iov[1].iov_base = local_notf_addr;
  iov[1].iov_len  = sizeof(gaspi_notification_t);

  struct fi_msg_rma rma_msg;

  rma_msg.msg_iov = iov;
  rma_msg.desc = desc;
  rma_msg.iov_count = 2;
  rma_msg.addr = fabric_ctx->io_fi_addr[queue][rank];

  rma_msg.rma_iov = rma_iov;
  rma_msg.rma_iov_count = 2;
  rma_msg.context = NULL;
  rma_msg.data = 0;

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;

  uint64_t flags = FI_FENCE;

  const size_t max_inject_size = fabric_ctx->info->tx_attr->inject_size;

  if (size < max_inject_size)
  {
    flags |= FI_INJECT;
  }

  ssize_t ret;
  do
  {
    ret = fi_writemsg (ep, &rma_msg, flags);

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a notify (%ld: %s).", ret, fi_strerror (-ret));

      return ret;
    }
  } while (ret);

  if (size >= max_inject_size)
  {
    gctx->ne_count_c[queue]++;
  }

  return GASPI_SUCCESS;
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
  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;
  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[rank];

  struct fi_tx_attr* tx_attr = fabric_ctx->info->tx_attr;

  if (tx_attr->rma_iov_limit > 1)
  {
    return pgaspi_dev_write_notify_with_rma_iov (gctx,
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

  return pgaspi_dev_write_notify_no_rma_iov (gctx,
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
    notifications_space_size - sizeof (gaspi_notification_t);

  uint64_t remote_addr =
    fabric_ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR ?
    (uint64_t) (remote_seg.notif_spc.addr + notifications_offset) :
    (uint64_t) notifications_offset;

  void* local_addr =
    (void*)local_seg.notif_spc.addr +
    notification_id * sizeof (gaspi_notification_t);

  struct fid_ep *ep = fabric_ctx->qC[queue]->ep;
  fi_addr_t dest = fabric_ctx->io_fi_addr[queue][rank];

  void* const desc = pgaspi_dev_get_mr_desc (gctx, local_seg.mr[1], rank);

  uint64_t key =
    pgaspi_ofi_is_local_fabric_avail (ofi_ctx, gctx->rank, rank) ?
    remote_seg.rkey[1].local :
    remote_seg.rkey[1].remote;

  int ret;
  do
  {
    ret = fi_read (ep,
                   local_addr,
                   sizeof (gaspi_notification_t),
                   desc,
                   dest,
                   remote_addr,
                   key,
                   NULL);

    if (ret && ret != -FI_EAGAIN)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Error initiating a read (%d: %s).", ret, fi_strerror (-ret));
      return ret;
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

  /* //TODO: wait or not? */
  /* /\* ret = pgaspi_dev_wait (gctx, queue, GASPI_BLOCK); *\/ */
  /* /\* if (ret != GASPI_SUCCESS) *\/ */
  /* /\* { *\/ */
  /* /\*   return ret; *\/ */
  /* /\* } *\/ */

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

  /* //TODO: wait or not? */
  /* ret = pgaspi_dev_wait (gctx, queue, GASPI_BLOCK); */
  /* if (ret != GASPI_SUCCESS) */
  /* { */
  /*   return ret; */
  /* } */

  return
    pgaspi_dev_remote_notify (gctx,
                              segment_id_notification,
                              segment_id_notification,
                              rank,
                              notification_id,
                              queue);
}
