/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2024

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
#include "GPI2.h"
#include "GPI2_IB.h"

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
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  enum ibv_send_flags sf;

  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  sf = (size > MAX_INLINE_BYTES)
    ? IBV_SEND_SIGNALED
    : IBV_SEND_SIGNALED | IBV_SEND_INLINE;

  slist.addr =
    (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
                 offset_local);

  slist.length = size;
  slist.lkey =
    ((struct ibv_mr *) gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.wr.rdma.remote_addr =
    (gctx->rrmd[segment_id_remote][rank].data.addr + offset_remote);

  swr.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[0];
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = sf;
  swr.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swr, &bad_wr))
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
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist;
  struct ibv_send_wr swr;

  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  slist.addr =
    (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
                 offset_local);

  slist.length = size;
  slist.lkey =
    ((struct ibv_mr *) gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].data.addr +
                             offset_remote);

  swr.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[0];
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_READ;
  swr.send_flags = IBV_SEND_SIGNALED;   // | IBV_SEND_FENCE;
  swr.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swr, &bad_wr))
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
  int ne = 0;
  struct ibv_wc wc;

  const int nr = gctx->ne_count_c[queue];
  const gaspi_cycles_t s0 = gaspi_get_cycles();

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  for (int i = 0; i < nr; i++)
  {
    do
    {
      ne = ibv_poll_cq (ib_dev_ctx->scqC[queue], 1, &wc);
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
  struct ibv_wc wc;

  const int nr = gctx->ne_count_c[queue];
  const gaspi_cycles_t s0 = gaspi_get_cycles();

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  int ne = 0;
  for (int i = 0; i < nr; i++)
  {
    do
    {
      ne = ibv_poll_cq (ib_dev_ctx->scqC[queue], 1, &wc);
      gctx->ne_count_c[queue] -= ne;    //TODO: this should be done below, when ne > 0

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


    if ((ne < 0) || (wc.status != IBV_WC_SUCCESS))
    {
      //TODO: for now here because we have to identify the rank
      // but should be out of device?
      gctx->state_vec[queue][wc.wr_id] = GASPI_STATE_CORRUPT;
      GASPI_DEBUG_PRINT_ERROR
        ("Failed request to %lu. Queue %d might be broken %s", wc.wr_id, queue,
         ibv_wc_status_str (wc.status));

      return GASPI_ERROR;
    }
  }

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
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[gctx->config->rw_list_elem_max];
  struct ibv_send_wr swr[gctx->config->rw_list_elem_max];

  if (gctx->ne_count_c[queue] + num > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_number_t entries_counter = 0;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    if (size[i] == 0)
    {
      continue;
    }

    slist[entries_counter].addr =
      (uintptr_t) (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
                   offset_local[i]);

    slist[entries_counter].length = size[i];
    slist[entries_counter].lkey =
      ((struct ibv_mr *) gctx->rrmd[segment_id_local[i]][gctx->rank].
       mr[0])->lkey;

    swr[entries_counter].wr.rdma.remote_addr =
      (gctx->rrmd[segment_id_remote[i]][rank].data.addr + offset_remote[i]);

    swr[entries_counter].wr.rdma.rkey = gctx->rrmd[segment_id_remote[i]][rank].rkey[0];
    swr[entries_counter].sg_list = &slist[entries_counter];
    swr[entries_counter].num_sge = 1;
    swr[entries_counter].wr_id = rank;
    swr[entries_counter].opcode = IBV_WR_RDMA_WRITE;
    swr[entries_counter].send_flags = IBV_SEND_SIGNALED;

    entries_counter++;
  }

  if (entries_counter > 0)
  {
    gaspi_number_t entry = 0;
    for (entry = 0; entry < entries_counter - 1; entry++)
    {
      swr[entry].next = &swr[entry + 1];
    }

    swr[entry].next = NULL;

    gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

    if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swr[0], &bad_wr))
    {
      return GASPI_ERROR;
    }

    gctx->ne_count_c[queue] += entries_counter;
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
                      gaspi_size_t * const size, const gaspi_queue_id_t queue)
{
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[gctx->config->rw_list_elem_max];
  struct ibv_send_wr swr[gctx->config->rw_list_elem_max];

  if (gctx->ne_count_c[queue] + num > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_number_t entries_counter = 0;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    if (size[i] == 0)
    {
      continue;
    }

    slist[entries_counter].addr =
      (uintptr_t) (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
                   offset_local[i]);

    slist[entries_counter].length = size[i];
    slist[entries_counter].lkey =
      ((struct ibv_mr *) gctx->rrmd[segment_id_local[i]][gctx->rank].
       mr[0])->lkey;

    swr[entries_counter].wr.rdma.remote_addr =
      (gctx->rrmd[segment_id_remote[i]][rank].data.addr + offset_remote[i]);

    swr[entries_counter].wr.rdma.rkey = gctx->rrmd[segment_id_remote[i]][rank].rkey[0];
    swr[entries_counter].sg_list = &slist[entries_counter];
    swr[entries_counter].num_sge = 1;
    swr[entries_counter].wr_id = rank;
    swr[entries_counter].opcode = IBV_WR_RDMA_READ;
    swr[entries_counter].send_flags = IBV_SEND_SIGNALED;

    entries_counter++;
  }

  if (entries_counter > 0)
  {
    gaspi_number_t entry = 0;
    for (entry = 0; entry < entries_counter - 1; entry++)
    {
      swr[entry].next = &swr[entry + 1];
    }

    swr[entry].next = NULL;

    gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

    if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swr[0], &bad_wr))
    {
      return GASPI_ERROR;
    }

    gctx->ne_count_c[queue] += entries_counter;
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
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slistN;
  struct ibv_send_wr swrN;

  if (gctx->ne_count_c[queue] == gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  slistN.addr =
    (uintptr_t) (gctx->nsrc.notif_spc.buf +
                 notification_id * sizeof (gaspi_notification_t));

  *((gaspi_notification_t *) slistN.addr) = notification_value;

  slistN.length = sizeof (gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *) gctx->nsrc.mr[1])->lkey;

  swrN.wr.rdma.remote_addr =
    (gctx->rrmd[segment_id_remote][rank].notif_spc.addr +
     notification_id * sizeof (gaspi_notification_t));
  swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[1];

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swrN.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swrN, &bad_wr))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue]++;

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
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;

  if (gctx->ne_count_c[queue] + 2 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  slist.addr =
    (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
                 offset_local);

  slist.length = size;
  slist.lkey =
    ((struct ibv_mr *) gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.wr.rdma.remote_addr = (gctx->rrmd[segment_id_remote][rank].data.addr +
                             offset_remote);

  swr.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[0];
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr =
    (uintptr_t) (gctx->nsrc.notif_spc.buf +
                 notification_id * sizeof (gaspi_notification_t));

  *((gaspi_notification_t *) slistN.addr) = notification_value;

  slistN.length = sizeof (gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *) gctx->nsrc.mr[1])->lkey;

  swrN.wr.rdma.remote_addr =
    (gctx->rrmd[segment_id_remote][rank].notif_spc.addr +
     notification_id * sizeof (gaspi_notification_t));
  swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[1];

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;;
  swrN.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swr, &bad_wr))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue] += 2;

  return GASPI_SUCCESS;
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
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[gctx->config->rw_list_elem_max], slistN;
  struct ibv_send_wr swr[gctx->config->rw_list_elem_max], swrN;

  if (gctx->ne_count_c[queue] + num + 1 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_number_t entries_counter = 0;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    if (size[i] == 0)
    {
      continue;
    }

    slist[entries_counter].addr =
      (uintptr_t)
      (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
       offset_local[i]);

    slist[entries_counter].length = size[i];
    slist[entries_counter].lkey =
      ((struct ibv_mr *)
       gctx->rrmd[segment_id_local[i]][gctx->rank].mr[0])->lkey;

    swr[entries_counter].wr.rdma.remote_addr =
      (gctx->rrmd[segment_id_remote[i]][rank].data.addr
       +
       offset_remote[i]);

    swr[entries_counter].wr.rdma.rkey =
      gctx->rrmd[segment_id_remote[i]][rank].rkey[0];

    swr[entries_counter].sg_list = &slist[entries_counter];
    swr[entries_counter].num_sge = 1;
    swr[entries_counter].wr_id = rank;
    swr[entries_counter].opcode = IBV_WR_RDMA_WRITE;
    swr[entries_counter].send_flags = IBV_SEND_SIGNALED;

    entries_counter++;
  }

  gaspi_number_t entry = 0;
  if (entries_counter > 0)
  {
    for (entry = 0; entry < entries_counter - 1; entry++)
    {
      swr[entry].next = &swr[entry + 1];

    }
  }
  swr[entry].next = &swrN;

  slistN.addr =
    (uintptr_t) (gctx->nsrc.notif_spc.buf +
                 notification_id * sizeof (gaspi_notification_t));

  *((gaspi_notification_t *) slistN.addr) = notification_value;

  slistN.length = sizeof (gaspi_notification_t);
  slistN.lkey = ((struct ibv_mr *) gctx->nsrc.mr[1])->lkey;

  swrN.wr.rdma.remote_addr =
    (gctx->rrmd[segment_id_notification][rank].notif_spc.addr +
     notification_id * sizeof (gaspi_notification_t));
  swrN.wr.rdma.rkey = gctx->rrmd[segment_id_notification][rank].rkey[1];

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swrN.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swr[0], &bad_wr))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue] += (int) entries_counter + 1;

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
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;

  if (gctx->ne_count_c[queue] + 2 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  slist.addr =
    (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].data.addr +
                 offset_local);

  slist.length = size;
  slist.lkey =
    ((struct ibv_mr *) gctx->rrmd[segment_id_local][gctx->rank].mr[0])->lkey;

  swr.wr.rdma.remote_addr =
    (gctx->rrmd[segment_id_remote][rank].data.addr + offset_remote);

  swr.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[0];
  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.wr_id = rank;
  swr.opcode = IBV_WR_RDMA_READ;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr =
    (uintptr_t) (gctx->rrmd[segment_id_local][gctx->rank].notif_spc.addr +
                 notification_id * sizeof (gaspi_notification_t));
  slistN.length = sizeof (gaspi_notification_t);
  slistN.lkey =
    ((struct ibv_mr *) gctx->rrmd[segment_id_local][gctx->rank].mr[1])->lkey;

  gaspi_number_t notifications_space_size = pgaspi_notifications_space_size ();
  swrN.wr.rdma.remote_addr =
    (gctx->rrmd[segment_id_remote][rank].notif_spc.addr
     + notifications_space_size
     - sizeof (gaspi_notification_t));
  swrN.wr.rdma.rkey = gctx->rrmd[segment_id_remote][rank].rkey[1];

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_READ;
  swrN.send_flags = IBV_SEND_SIGNALED;  // | IBV_SEND_FENCE;;
  swrN.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swr, &bad_wr))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue] += 2;

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
  struct ibv_send_wr *bad_wr;
  struct ibv_sge slist[gctx->config->rw_list_elem_max], slistN;
  struct ibv_send_wr swr[gctx->config->rw_list_elem_max], swrN;

  if (gctx->ne_count_c[queue] + num + 1 > gctx->config->queue_size_max)
  {
    return GASPI_QUEUE_FULL;
  }

  gaspi_number_t entries_counter = 0;

  for (gaspi_number_t i = 0; i < num; i++)
  {
    if (size[i] == 0)
    {
      continue;
    }

    slist[entries_counter].addr =
      (uintptr_t)
      (gctx->rrmd[segment_id_local[i]][gctx->rank].data.addr +
       offset_local[i]);

    slist[entries_counter].length = size[i];
    slist[entries_counter].lkey =
      ((struct ibv_mr *)
       gctx->rrmd[segment_id_local[i]][gctx->rank]
       .mr[0])->lkey;

    swr[entries_counter].wr.rdma.remote_addr =
      (gctx->rrmd[segment_id_remote[i]][rank].data.addr
       +
       offset_remote[i]);

    swr[entries_counter].wr.rdma.rkey =
      gctx->rrmd[segment_id_remote[i]][rank].rkey[0];

    swr[entries_counter].sg_list = &slist[entries_counter];
    swr[entries_counter].num_sge = 1;
    swr[entries_counter].wr_id = rank;
    swr[entries_counter].opcode = IBV_WR_RDMA_READ;
    swr[entries_counter].send_flags = IBV_SEND_SIGNALED;

    entries_counter++;
  }

  gaspi_number_t entry = 0;
  if (entries_counter > 0)
  {
    for (entry = 0; entry < entries_counter - 1; entry++)
    {
      swr[entry].next = &swr[entry + 1];

    }
  }
  swr[entry].next = &swrN;

  slistN.addr =
    (uintptr_t) (gctx->rrmd[segment_id_notification][gctx->rank].
                 notif_spc.addr +
                 notification_id * sizeof (gaspi_notification_t));
  slistN.length = sizeof (gaspi_notification_t);
  slistN.lkey =
    ((struct ibv_mr *) gctx->rrmd[segment_id_notification][gctx->rank].
     mr[1])->lkey;

  gaspi_number_t notifications_space_size = pgaspi_notifications_space_size ();
  swrN.wr.rdma.remote_addr =
    (gctx->rrmd[segment_id_notification][rank].notif_spc.addr
     + notifications_space_size
     - sizeof (gaspi_notification_t));
  swrN.wr.rdma.rkey = gctx->rrmd[segment_id_notification][rank].rkey[1];

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.wr_id = rank;
  swrN.opcode = IBV_WR_RDMA_READ;
  swrN.send_flags = IBV_SEND_SIGNALED;  // | IBV_SEND_FENCE;
  swrN.next = NULL;

  gaspi_ib_ctx *const ib_dev_ctx = (gaspi_ib_ctx *) gctx->device->ctx;

  if (ibv_post_send (ib_dev_ctx->qpC[queue][rank], &swr[0], &bad_wr))
  {
    return GASPI_ERROR;
  }

  gctx->ne_count_c[queue] += (int) entries_counter + 1;

  return GASPI_SUCCESS;
}
