/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2018

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
#include <stddef.h>
#include <sys/timeb.h>
#include <unistd.h>

#include "PGASPI.h"
#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_Utility.h"
#include "GPI2_SN.h"
#include "GPI2_SEG.h"

#pragma weak gaspi_segment_max = pgaspi_segment_max
gaspi_return_t
pgaspi_segment_max (gaspi_number_t * const segment_max)
{
  gaspi_verify_null_ptr(segment_max);
  gaspi_verify_init("gaspi_segment_max");

  *segment_max = GASPI_MAX_MSEGS;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_size = pgaspi_segment_size
gaspi_return_t
pgaspi_segment_size (const gaspi_segment_id_t segment_id,
		     const gaspi_rank_t rank,
		     gaspi_size_t * const size)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_segment_size");
  gaspi_verify_segment(segment_id);
  gaspi_verify_null_ptr(gctx->rrmd[segment_id]);
  gaspi_verify_null_ptr(size);

  gaspi_size_t seg_size = gctx->rrmd[segment_id][rank].size;

  gaspi_verify_segment_size(seg_size);

  *size = seg_size;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_ptr = pgaspi_segment_ptr
gaspi_return_t
pgaspi_segment_ptr (const gaspi_segment_id_t segment_id, gaspi_pointer_t * ptr)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_segment_ptr");
  gaspi_verify_segment(segment_id);
  gaspi_verify_null_ptr(gctx->rrmd[segment_id]);
  gaspi_verify_null_ptr(ptr);

  gaspi_verify_segment_size(gctx->rrmd[segment_id][gctx->rank].size);

  *ptr = gctx->rrmd[segment_id][gctx->rank].data.buf;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_list = pgaspi_segment_list
gaspi_return_t
pgaspi_segment_list (const gaspi_number_t num,
		     gaspi_segment_id_t * const segment_id_list)
{
  int i, idx = 0;
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_segment_list");
  gaspi_verify_null_ptr(segment_id_list);

  for(i = 0; i < GASPI_MAX_MSEGS; i++)
    {
      if( gctx->rrmd[(gaspi_segment_id_t) i] != NULL )
	{
	  if( gctx->rrmd[(gaspi_segment_id_t) i][gctx->rank].trans )
	    {
	      segment_id_list[idx++] = (gaspi_segment_id_t) i;
	    }
	}
    }

  if( idx != gctx->mseg_cnt )
    {
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_num = pgaspi_segment_num
gaspi_return_t
pgaspi_segment_num (gaspi_number_t * const segment_num)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_segment_num");
  gaspi_verify_null_ptr(segment_num);

  *segment_num = (gaspi_number_t) gctx->mseg_cnt;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_avail_local = pgaspi_segment_avail_local
gaspi_return_t
pgaspi_segment_avail_local (gaspi_segment_id_t * const avail_seg_id)
{
  gaspi_verify_null_ptr(avail_seg_id);

  gaspi_number_t num_segs;
  if( gaspi_segment_num (&num_segs) != GASPI_SUCCESS)
    {
      return GASPI_ERROR;
    }

  if( num_segs == 0 )
    {
      *avail_seg_id = 0;
      return GASPI_SUCCESS;
    }

  gaspi_number_t segs_max;
  if( gaspi_segment_max (&segs_max) != GASPI_SUCCESS )
    {
      return GASPI_ERROR;
    }

  if( num_segs == segs_max )
    {
      return GASPI_ERR_MANY_SEG;
    }

  gaspi_segment_id_t *segment_ids = malloc(num_segs * sizeof(gaspi_segment_id_t));
  if( segment_ids == NULL )
    {
      return GASPI_ERR_MEMALLOC;
    }

  if( gaspi_segment_list (num_segs, segment_ids) != GASPI_SUCCESS )
    {
      free(segment_ids);
      return GASPI_ERROR;
    }

  gaspi_number_t i;
  for(i = 1; i < num_segs; i++)
    {
      if( segment_ids[i] != segment_ids[i-1]+1 )
	{
	  *avail_seg_id = (gaspi_segment_id_t) i ;
	  free(segment_ids);
	  return GASPI_SUCCESS;
	}
    }
  *avail_seg_id = num_segs;

  free(segment_ids);

  return GASPI_SUCCESS;
}

static inline int
pgaspi_segment_create_desc( gaspi_context_t * const gctx,
			    const gaspi_segment_id_t segment_id)
{
  if( gctx->rrmd[segment_id] == NULL)
    {
      gctx->rrmd[segment_id] = (gaspi_rc_mseg_t *) calloc (gctx->tnc, sizeof (gaspi_rc_mseg_t));

      if( gctx->rrmd[segment_id] == NULL)
	{
	  return 1;
	}
    }

  return 0;
}

#pragma weak gaspi_segment_alloc = pgaspi_segment_alloc
gaspi_return_t
pgaspi_segment_alloc (const gaspi_segment_id_t segment_id,
		      const gaspi_size_t size,
		      const gaspi_alloc_t alloc_policy)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_segment_alloc");
  gaspi_verify_segment_size(size);
  gaspi_verify_segment(segment_id);

  if( gctx->mseg_cnt >= GASPI_MAX_MSEGS )
    {
      return GASPI_ERR_MANY_SEG;
    }

  lock_gaspi_tout (&(gctx->mseg_lock), GASPI_BLOCK);

  gaspi_return_t eret = GASPI_ERROR;

  /*  TODO: for now like this, but we need to change this */
#ifndef GPI2_CUDA
  if( pgaspi_segment_create_desc(gctx, segment_id) != 0)
    {
      eret = GASPI_ERR_MEMALLOC;
      goto endL;
    }

  /* Already exists?*/
  /* TODO: not really the right way */
  if( gctx->rrmd[segment_id][gctx->rank].size )
    {
      eret = GASPI_SUCCESS;
      goto endL;
    }

  if( pgaspi_alloc_page_aligned(&(gctx->rrmd[segment_id][gctx->rank].data.ptr), size + NOTIFY_OFFSET ) != 0 )
    {
      gaspi_print_error ("Memory allocation (posix_memalign) failed");
      eret = GASPI_ERR_MEMALLOC;
      goto endL;
    }

  memset(gctx->rrmd[segment_id][gctx->rank].data.ptr, 0, NOTIFY_OFFSET);

  if( GASPI_MEM_INITIALIZED == alloc_policy )
    {
      memset (gctx->rrmd[segment_id][gctx->rank].data.ptr, 0, size + NOTIFY_OFFSET);
    }

  gctx->rrmd[segment_id][gctx->rank].size = size;
  gctx->rrmd[segment_id][gctx->rank].notif_spc_size = NOTIFY_OFFSET;
  gctx->rrmd[segment_id][gctx->rank].notif_spc.addr = gctx->rrmd[segment_id][gctx->rank].data.addr;
  gctx->rrmd[segment_id][gctx->rank].data.addr += NOTIFY_OFFSET;
  gctx->rrmd[segment_id][gctx->rank].user_provided = 0;

  if( pgaspi_dev_register_mem(gctx, &(gctx->rrmd[segment_id][gctx->rank])) < 0 )
    {
      free(gctx->rrmd[segment_id][gctx->rank].data.ptr);
      goto endL;
    }

  /* set fixed notification value ( =1) for read_notify */
  unsigned char *segPtr = (unsigned char *) gctx->rrmd[segment_id][gctx->rank].notif_spc.addr + NOTIFY_OFFSET - sizeof(gaspi_notification_t);
  gaspi_notification_t *p = (gaspi_notification_t *) segPtr;
  *p = 1;

#else
  eret = pgaspi_dev_segment_alloc(segment_id, size, alloc_policy);
  if( eret != GASPI_SUCCESS )
    goto endL;
#endif /* GPI2_CUDA */

  gctx->mseg_cnt++;

  eret = GASPI_SUCCESS;

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_SEG_ALLOC, 1);

 endL:
  unlock_gaspi (&(gctx->mseg_lock));
  return eret;

}

#pragma weak gaspi_segment_delete = pgaspi_segment_delete
gaspi_return_t
pgaspi_segment_delete (const gaspi_segment_id_t segment_id)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_segment_delete");
  gaspi_verify_segment(segment_id);
  gaspi_verify_null_ptr(gctx->rrmd[segment_id]);

  gaspi_verify_segment_size(gctx->rrmd[segment_id][gctx->rank].size);

  gaspi_return_t eret = GASPI_ERROR;

  lock_gaspi_tout(&(gctx->mseg_lock), GASPI_BLOCK);

  /*  TODO: for now like this but we need a better solution */
#ifdef GPI2_CUDA
  eret = pgaspi_dev_segment_delete(segment_id);
#else
  if(pgaspi_dev_unregister_mem(gctx, &(gctx->rrmd[segment_id][gctx->rank])) < 0)
    {
      unlock_gaspi (&(gctx->mseg_lock));
      return GASPI_ERR_DEVICE;
    }

  /* For both "normal" and user-provided segments, the notif_spc
     points to begin of memory and only the size changes.
  */
  free (gctx->rrmd[segment_id][gctx->rank].notif_spc.buf);

  gctx->rrmd[segment_id][gctx->rank].data.buf = NULL;
  gctx->rrmd[segment_id][gctx->rank].notif_spc.buf = NULL;
  gctx->rrmd[segment_id][gctx->rank].size = 0;
  gctx->rrmd[segment_id][gctx->rank].notif_spc_size = 0;
  gctx->rrmd[segment_id][gctx->rank].trans = 0;
  gctx->rrmd[segment_id][gctx->rank].mr[0] = NULL;
  gctx->rrmd[segment_id][gctx->rank].mr[1] = NULL;
#ifdef GPI2_DEVICE_IB
  gctx->rrmd[segment_id][gctx->rank].rkey[0] = 0;
  gctx->rrmd[segment_id][gctx->rank].rkey[1] = 0;
#endif
  gctx->rrmd[segment_id][gctx->rank].user_provided = 0;

  /* Reset trans info flag for all ranks */
  int r;
  for(r = 0; r < gctx->tnc; r++)
    {
      gctx->rrmd[segment_id][r].trans = 0;
    }

  eret = GASPI_SUCCESS;
#endif

  gctx->mseg_cnt--;

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_SEG_DELETE, 1);

  unlock_gaspi (&(gctx->mseg_lock));

  return eret;
}

#pragma weak gaspi_segment_register = pgaspi_segment_register
gaspi_return_t
pgaspi_segment_register(const gaspi_segment_id_t segment_id,
			const gaspi_rank_t rank,
			const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_segment_register");
  gaspi_verify_segment(segment_id);
  gaspi_verify_null_ptr(gctx->rrmd[segment_id]);
  gaspi_verify_rank(rank);

  gaspi_verify_segment_size(gctx->rrmd[segment_id][gctx->rank].size);

  if( rank == gctx->rank )
    {
      gctx->rrmd[segment_id][rank].trans = 1;
      return GASPI_SUCCESS;
    }

  if( lock_gaspi_tout(&(gctx->ctx_lock), timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  gaspi_return_t eret = gaspi_sn_command(GASPI_SN_SEG_REGISTER, rank, timeout_ms, (void *) &segment_id);

  gctx->rrmd[segment_id][rank].trans = 1;

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_SEG_REGISTER, 1);

  unlock_gaspi(&(gctx->ctx_lock));

  return eret;
}

//TODO: need a better name
int
gaspi_segment_set(const gaspi_segment_descriptor_t snp)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( !(gctx->dev_init) )
    {
      return -1;
    }

  if( snp.seg_id < 0 || snp.seg_id >= GASPI_MAX_MSEGS )
    {
      return -1;
    }

  lock_gaspi_tout(&(gctx->mseg_lock), GASPI_BLOCK);

  //TODO: use segment_create_desc
  if( gctx->rrmd[snp.seg_id] == NULL )
    {
      gctx->rrmd[snp.seg_id] = (gaspi_rc_mseg_t *) calloc (gctx->tnc, sizeof (gaspi_rc_mseg_t));

      if( gctx->rrmd[snp.seg_id] == NULL )
	{
	  unlock_gaspi(&(gctx->mseg_lock));
	  return -1;
	}
    }

  /* TODO: don't allow re-registration? */
  /* for now we allow re-registration */
  /* if(gctx->rrmd[snp.seg_id][snp.rem_rank].size) -> re-registration error case */
  gctx->rrmd[snp.seg_id][snp.rank].data.addr = snp.addr;
  gctx->rrmd[snp.seg_id][snp.rank].notif_spc.addr = snp.notif_addr;
  gctx->rrmd[snp.seg_id][snp.rank].size = snp.size;

#ifdef GPI2_DEVICE_IB
  gctx->rrmd[snp.seg_id][snp.rank].rkey[0] = snp.rkey[0];
  gctx->rrmd[snp.seg_id][snp.rank].rkey[1] = snp.rkey[1];
#endif

#ifdef GPI2_CUDA
  gctx->rrmd[snp.seg_id][snp.rank].host_rkey = snp.host_rkey;
  gctx->rrmd[snp.seg_id][snp.rank].host_addr = snp.host_addr;

  if(snp.host_addr != 0 )
    gctx->rrmd[snp.seg_id][snp.rank].cuda_dev_id = 1;
  else
    gctx->rrmd[snp.seg_id][snp.rank].cuda_dev_id = -1;
#endif

  unlock_gaspi(&(gctx->mseg_lock));
  return 0;
}

static gaspi_return_t
pgaspi_segment_register_group(gaspi_context_t * const gctx,
			      const gaspi_segment_id_t segment_id,
			      const gaspi_group_t group,
			      const gaspi_timeout_t timeout_ms)
{
  if( lock_gaspi_tout(&(gctx->ctx_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  //prepare my segment info
  gaspi_segment_descriptor_t cdh;
  memset(&cdh, 0, sizeof(cdh));

  gaspi_rc_mseg_t * const mseg_info = &(gctx->rrmd[segment_id][gctx->rank]);

  cdh.rank = gctx->rank;
  cdh.seg_id = segment_id;
  cdh.addr = mseg_info->data.addr;
  cdh.notif_addr = mseg_info->notif_spc.addr;
  cdh.size = mseg_info->size;

#ifdef GPI2_CUDA
  cdh.host_rkey = mseg_info->host_rkey;
  cdh.host_addr = mseg_info->host_addr;
#endif

#ifdef GPI2_DEVICE_IB
  cdh.rkey[0] = mseg_info->rkey[0];
  cdh.rkey[1] = mseg_info->rkey[1];
#endif

  gaspi_segment_descriptor_t* result = calloc(gctx->groups[group].tnc, sizeof(gaspi_segment_descriptor_t));
  if( result == NULL )
    {
      unlock_gaspi(&(gctx->ctx_lock));
      return GASPI_ERR_MEMALLOC;
    }

  if( gaspi_sn_allgather(gctx, &cdh, result, sizeof(gaspi_segment_descriptor_t), group, timeout_ms) != 0 )
    {
      free(result);
      unlock_gaspi(&(gctx->ctx_lock));
      return GASPI_ERROR;
    }

  int r;
  for(r = 0; r < gctx->groups[group].tnc; r++)
    {
      if( gaspi_segment_set(result[r]) < 0 )
	{
	  free(result);
	  unlock_gaspi(&(gctx->ctx_lock));
	  return GASPI_ERROR;
	}

      gctx->rrmd[segment_id][r].trans = 1;
    }

  free(result);

  unlock_gaspi(&(gctx->ctx_lock));

  return GASPI_SUCCESS;
}

/* TODO: from the spec: */
/* 1) connect all in group => maybe remove this from spec instead ?*/
/* 2) Creating a new segment with an existing segment ID results in
   undefined behavior */
#pragma weak gaspi_segment_create = pgaspi_segment_create
gaspi_return_t
pgaspi_segment_create(const gaspi_segment_id_t segment_id,
		      const gaspi_size_t size,
		      const gaspi_group_t group,
		      const gaspi_timeout_t timeout_ms,
		      const gaspi_alloc_t alloc_policy)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_group(group);

  gaspi_return_t eret = pgaspi_segment_alloc (segment_id, size, alloc_policy);
  if( eret != GASPI_SUCCESS )
    {
      return eret;
    }
  eret = pgaspi_segment_register_group(gctx, segment_id, group, timeout_ms);
  if( eret != GASPI_SUCCESS )
    {
      unlock_gaspi(&(gctx->ctx_lock));
      return eret;
    }

  if( GASPI_TOPOLOGY_STATIC == gctx->config->build_infrastructure )
    {
      int r;
      for(r = gctx->groups[group].rank; r < gctx->groups[group].tnc; r++)
	{
	  eret = pgaspi_connect(gctx->groups[group].rank_grp[r], timeout_ms);
	  if( eret != GASPI_SUCCESS )
	    {
	      return eret;
	    }
	}
    }

  eret = pgaspi_barrier(group, timeout_ms);

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_SEG_CREATE, 1);

  return eret;
}

/* Extensions */
/* TODO: */
/* - GPU case */
/* - merge common/repetead code from other segment related function (create, alloc, ...) */
/* - check/deal with alignment issues */
#pragma weak gaspi_segment_bind = pgaspi_segment_bind
gaspi_return_t
pgaspi_segment_bind ( gaspi_segment_id_t const segment_id,
		      gaspi_pointer_t const pointer,
		      gaspi_size_t const size,
		      gaspi_memory_description_t const memory_description)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_segment_bind");
  gaspi_verify_segment_size(size);
  gaspi_verify_segment(segment_id);

  const int myrank = (int) gctx->rank;

  if( gctx->mseg_cnt >= GASPI_MAX_MSEGS )
    {
      return GASPI_ERR_MANY_SEG;
    }

  lock_gaspi_tout (&(gctx->mseg_lock), GASPI_BLOCK);

  gaspi_return_t eret = GASPI_ERROR;

  if( pgaspi_segment_create_desc(gctx, segment_id) != 0)
    {
      eret = GASPI_ERR_MEMALLOC;
      goto endL;
    }

  if( gctx->rrmd[segment_id][myrank].size )
    {
      eret = GASPI_SUCCESS;
      goto endL;
    }

  /* Get space for notifications and register it as well */
  long page_size = sysconf (_SC_PAGESIZE);

  if( page_size < 0 )
    {
      gaspi_print_error ("Failed to get system's page size.");
      goto endL;
    }

  if( posix_memalign( (void **) &gctx->rrmd[segment_id][myrank].notif_spc.ptr,
		      page_size,
		      NOTIFY_OFFSET) != 0 )
    {
      gaspi_print_error ("Memory allocation failed.");
      eret = GASPI_ERR_MEMALLOC;
      goto endL;
    }

  memset (gctx->rrmd[segment_id][myrank].notif_spc.ptr, 0, NOTIFY_OFFSET);

  /* Set the segment data pointer and size */
  gctx->rrmd[segment_id][myrank].data.ptr = pointer;
  gctx->rrmd[segment_id][myrank].size = size;

  gctx->rrmd[segment_id][myrank].notif_spc_size = NOTIFY_OFFSET;

  gctx->rrmd[segment_id][gctx->rank].user_provided = 1;

  /* TODO: what to do with the memory description?? */
  gctx->rrmd[segment_id][myrank].desc = memory_description;

  /* Register segment with the device */
  if( pgaspi_dev_register_mem( gctx, &(gctx->rrmd[segment_id][myrank])) < 0)
    {
      eret = GASPI_ERR_DEVICE;
      goto endL;
    }

  /* set fixed notification value ( =1) for read_notify */
  unsigned char *segPtr = (unsigned char *) gctx->rrmd[segment_id][gctx->rank].notif_spc.addr + NOTIFY_OFFSET - sizeof(gaspi_notification_t);
  gaspi_notification_t *p = (gaspi_notification_t *) segPtr;
  *p = 1;

  gctx->mseg_cnt++;

  eret = GASPI_SUCCESS;

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_SEG_BIND, 1);

 endL:
  unlock_gaspi (&(gctx->mseg_lock));
  return eret;
}

#pragma weak gaspi_segment_use = pgaspi_segment_use
gaspi_return_t
pgaspi_segment_use ( gaspi_segment_id_t const segment_id,
		     gaspi_pointer_t const pointer,
		     gaspi_size_t const size,
		     gaspi_group_t const group,
		     gaspi_timeout_t const timeout,
		     gaspi_memory_description_t const memory_description)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_return_t ret = pgaspi_segment_bind(segment_id, pointer, size, memory_description);
  if( GASPI_SUCCESS != ret )
    {
      return ret;
    }

  gaspi_return_t eret = pgaspi_segment_register_group(gctx, segment_id, group, timeout);
  if( eret != GASPI_SUCCESS )
    {
      return eret;
    }

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_SEG_USE, 1);

  return gaspi_barrier( group, timeout);
}
