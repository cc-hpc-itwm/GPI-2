/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2019

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
#include "GASPI.h"
#include "GPI2_TCP.h"

/* Communication functions */
gaspi_return_t
pgaspi_dev_write (gaspi_context_t * const gctx,
                  const gaspi_segment_id_t segment_id_local,
                  const gaspi_offset_t offset_local,
                  const gaspi_rank_t rank,
                  const gaspi_segment_id_t segment_id_remote,
                  const gaspi_offset_t offset_remote,
                  const gaspi_size_t size, const gaspi_queue_id_t queue)
{
  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  tcp_dev_wr_t wr =
    {
      .wr_id = rank,
      .cq_handle = tcp_dev_ctx->scqC[queue]->num,
      .source = gctx->rank,
      .target = rank,
      .local_addr =
      (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
                   offset_local),
      .remote_addr =
      (gctx->rrmd[segment_id_remote][rank].data.addr + offset_remote),
      .length = size,
      .swap = 0,
      .compare_add = 0,
      .opcode = POST_RDMA_WRITE
    };

  if (write (tcp_dev_ctx->qpC[queue]->handle, &wr, sizeof (tcp_dev_wr_t)) <
      (ssize_t) sizeof (tcp_dev_wr_t))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_read (gaspi_context_t * const gctx,
                 const gaspi_segment_id_t segment_id_local,
                 const gaspi_offset_t offset_local,
                 const gaspi_rank_t rank,
                 const gaspi_segment_id_t segment_id_remote,
                 const gaspi_offset_t offset_remote,
                 const gaspi_size_t size, const gaspi_queue_id_t queue)
{
  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  tcp_dev_wr_t wr =
    {
      .wr_id = rank,
      .cq_handle = tcp_dev_ctx->scqC[queue]->num,
      .source = gctx->rank,
      .target = rank,
      .local_addr =
      (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
                   offset_local),
      .remote_addr =
      (gctx->rrmd[segment_id_remote][rank].data.addr + offset_remote),
      .length = size,
      .swap = 0,
      .compare_add = 0,
      .opcode = POST_RDMA_READ
    };

  if (write (tcp_dev_ctx->qpC[queue]->handle, &wr, sizeof (tcp_dev_wr_t)) <
      (ssize_t) sizeof (tcp_dev_wr_t))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_purge (gaspi_context_t * const gctx,
                  const gaspi_queue_id_t queue,
                  const gaspi_timeout_t timeout_ms)
{
  tcp_dev_wc_t wc;
  const int nr = gctx->ne_count_c[queue];

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  int ne = 0;
  for (int i = 0; i < nr; i++)
  {
    do
    {
      ne = tcp_dev_return_wc (tcp_dev_ctx->scqC[queue], &wc);
      gctx->ne_count_c[queue] -= ne;

      if (ne == 0)
      {
        const gaspi_cycles_t s1 = gaspi_get_cycles();
        const gaspi_cycles_t tdelta = s1 - s0;

        const float ms = (float) tdelta * gctx->cycles_to_msecs;

        if (ms > timeout_ms)
        {
          return GASPI_TIMEOUT;
        }
      }
    }
    while (ne == 0);
  }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_wait (gaspi_context_t * const gctx,
                 const gaspi_queue_id_t queue,
                 const gaspi_timeout_t timeout_ms)
{
  tcp_dev_wc_t wc;

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  const int nr = gctx->ne_count_c[queue];
  const gaspi_cycles_t s0 = gaspi_get_cycles();

  int ne = 0;
  for (int i = 0; i < nr; i++)
  {
    do
    {
      ne = tcp_dev_return_wc (tcp_dev_ctx->scqC[queue], &wc);
      gctx->ne_count_c[queue] -= ne;

      if (ne == 0)
      {
        const gaspi_cycles_t s1 = gaspi_get_cycles();
        const gaspi_cycles_t tdelta = s1 - s0;

        const float ms = (float) tdelta * gctx->cycles_to_msecs;

        if (ms > timeout_ms)
        {
          return GASPI_TIMEOUT;
        }
      }
    }
    while (ne == 0);

    if ((ne < 0) || (wc.status != TCP_WC_SUCCESS))
    {
      gctx->state_vec[queue][wc.wr_id] = GASPI_STATE_CORRUPT;
      return GASPI_ERROR;
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

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  gaspi_notification_t *not_val_ptr =
    (gaspi_notification_t *) malloc (sizeof (notification_value));
  *not_val_ptr = notification_value;

  tcp_dev_wr_t wr =
    {
      .wr_id = rank,
      .cq_handle = tcp_dev_ctx->scqC[queue]->num,
      .source = gctx->rank,
      .target = rank,
      .local_addr = (uintptr_t) not_val_ptr,
      .remote_addr =
      (gctx->rrmd[segment_id_remote][rank].notif_spc.addr +
       notification_id * sizeof (gaspi_notification_t)),
      .length = sizeof (notification_value),
      .swap = 0,
      .opcode = POST_RDMA_WRITE_INLINED
    };

  if (write (tcp_dev_ctx->qpC[queue]->handle, &wr, sizeof (tcp_dev_wr_t)) <
      (ssize_t) sizeof (tcp_dev_wr_t))
  {
    return GASPI_ERROR;
  }

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
                       gaspi_size_t * const size, const gaspi_queue_id_t queue)
{
  if (gctx->ne_count_c[queue] + num > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    tcp_dev_wr_t wr = {
      .wr_id = rank,
      .cq_handle = tcp_dev_ctx->scqC[queue]->num,
      .source = gctx->rank,
      .target = rank,
      .local_addr =
        (uintptr_t) (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
                     offset_local[i]),
      .remote_addr =
        (gctx->rrmd[segment_id_remote[i]][rank].data.addr + offset_remote[i]),
      .length = size[i],
      .swap = 0,
      .compare_add = 0,
      .opcode = POST_RDMA_WRITE
    };

    if (write (tcp_dev_ctx->qpC[queue]->handle, &wr, sizeof (tcp_dev_wr_t)) <
        (ssize_t) sizeof (tcp_dev_wr_t))
    {
      return GASPI_ERROR;
    }
  }

  gctx->ne_count_c[queue] += num;

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
                      gaspi_size_t * const size, const gaspi_queue_id_t queue)
{
  if (gctx->ne_count_c[queue] + num > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }
  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    tcp_dev_wr_t wr =
      {
        .wr_id = rank,
        .cq_handle = tcp_dev_ctx->scqC[queue]->num,
        .source = gctx->rank,
        .target = rank,
        .local_addr =
        (uintptr_t) (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
                     offset_local[i]),
        .remote_addr =
        (gctx->rrmd[segment_id_remote[i]][rank].data.addr + offset_remote[i]),
        .length = size[i],
        .swap = 0,
        .compare_add = 0,
        .opcode = POST_RDMA_READ
      };

    if (write (tcp_dev_ctx->qpC[queue]->handle, &wr, sizeof (tcp_dev_wr_t)) <
        (ssize_t) sizeof (tcp_dev_wr_t))
    {
      return GASPI_ERROR;
    }
  }

  gctx->ne_count_c[queue] += num;

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
  if (gctx->ne_count_c[queue] + 2 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_return_t ret = pgaspi_dev_write (gctx,
                                         segment_id_local, offset_local, rank,
                                         segment_id_remote, offset_remote,
                                         size,
                                         queue);

  if (ret != GASPI_SUCCESS)
  {
    return ret;
  }

  return pgaspi_dev_notify (gctx, segment_id_remote, rank, notification_id,
                            notification_value, queue);
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
  if (gctx->ne_count_c[queue] + num + 1 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  //TODO: check different with and without extra function calls
  gaspi_return_t ret = pgaspi_dev_write_list (gctx,
                                              num, segment_id_local,
                                              offset_local, rank,
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
  if (gctx->ne_count_c[queue] + 2 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_return_t ret = pgaspi_dev_read (gctx,
                                        segment_id_local, offset_local, rank,
                                        segment_id_remote, offset_remote, size,
                                        queue);

  if (ret != GASPI_SUCCESS)
  {
    return ret;
  }

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  //notification
  tcp_dev_wr_t wr =
    {
      .wr_id = rank,
      .cq_handle = tcp_dev_ctx->scqC[queue]->num,
      .source = gctx->rank,
      .target = rank,
      .local_addr =
      (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].notif_spc.addr +
                   notification_id * sizeof (gaspi_notification_t)),
      .remote_addr =
        (gctx->rrmd[segment_id_remote][rank].notif_spc.addr
         + NOTIFICATIONS_SPACE_SIZE
         - sizeof (gaspi_notification_t)),
      .length = sizeof (gaspi_notification_t),
      .swap = 0,
      .opcode = POST_RDMA_READ
    };

  if (write (tcp_dev_ctx->qpC[queue]->handle, &wr, sizeof (tcp_dev_wr_t)) <
      (ssize_t) sizeof (tcp_dev_wr_t))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
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
  if (gctx->ne_count_c[queue] + num + 1 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_return_t ret = pgaspi_dev_read_list (gctx,
                                             num, segment_id_local,
                                             offset_local, rank,
                                             segment_id_remote, offset_remote,
                                             size,
                                             queue);

  if (ret != GASPI_SUCCESS)
  {
    return ret;
  }

  gaspi_tcp_ctx *const tcp_dev_ctx = (gaspi_tcp_ctx *) gctx->device->ctx;

  //notification
  tcp_dev_wr_t wr =
    {
      .wr_id = rank,
      .cq_handle = tcp_dev_ctx->scqC[queue]->num,
      .source = gctx->rank,
      .target = rank,
      .local_addr =
      (uintptr_t) (gctx->rrmd[segment_id_notification][gctx->rank].notif_spc.
                   addr + notification_id * sizeof (gaspi_notification_t)),
      .remote_addr =
        (gctx->rrmd[segment_id_notification][rank].notif_spc.addr
         + NOTIFICATIONS_SPACE_SIZE
         - sizeof (gaspi_notification_t)),
      .length = sizeof (gaspi_notification_t),
      .swap = 0,
      .opcode = POST_RDMA_READ
    };

  if (write (tcp_dev_ctx->qpC[queue]->handle, &wr, sizeof (tcp_dev_wr_t)) <
      (ssize_t) sizeof (tcp_dev_wr_t))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue]++;

  return GASPI_SUCCESS;
}
