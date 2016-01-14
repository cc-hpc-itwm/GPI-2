/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2015

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

#pragma weak gaspi_segment_max = pgaspi_segment_max
gaspi_return_t
pgaspi_segment_max (gaspi_number_t * const segment_max)
{
  gaspi_verify_null_ptr(segment_max);

  *segment_max = GASPI_MAX_MSEGS;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_size = pgaspi_segment_size
gaspi_return_t
pgaspi_segment_size (const gaspi_segment_id_t segment_id,
		     const gaspi_rank_t rank,
		     gaspi_size_t * const size)
{
  gaspi_verify_init("gaspi_segment_size");
  gaspi_verify_segment(segment_id);
  gaspi_verify_null_ptr(glb_gaspi_ctx.rrmd[segment_id]);
  gaspi_verify_null_ptr(size);

  gaspi_size_t seg_size = glb_gaspi_ctx.rrmd[segment_id][rank].size;

  gaspi_verify_segment_size(seg_size);

  *size = seg_size;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_ptr = pgaspi_segment_ptr
gaspi_return_t
pgaspi_segment_ptr (const gaspi_segment_id_t segment_id, gaspi_pointer_t * ptr)
{
  gaspi_verify_init("gaspi_segment_ptr");
  gaspi_verify_segment(segment_id);
  gaspi_verify_null_ptr(glb_gaspi_ctx.rrmd[segment_id]);
  gaspi_verify_null_ptr(ptr);

  gaspi_verify_segment_size(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size);

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId >= 0)
    *ptr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.buf;
  else
#endif

    *ptr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.buf + NOTIFY_OFFSET;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_list = pgaspi_segment_list
gaspi_return_t
pgaspi_segment_list (const gaspi_number_t num,
		     gaspi_segment_id_t * const segment_id_list)
{
  int i, idx = 0;

  gaspi_verify_init("gaspi_segment_list");
  gaspi_verify_null_ptr(segment_id_list);

  for (i = 0; i < GASPI_MAX_MSEGS; i++)
    {
      if(glb_gaspi_ctx.rrmd[(gaspi_segment_id_t) i] != NULL )
	if(glb_gaspi_ctx.rrmd[(gaspi_segment_id_t) i][glb_gaspi_ctx.rank].trans)
	  segment_id_list[idx++] = (gaspi_segment_id_t) i;
    }

  if (idx != glb_gaspi_ctx.mseg_cnt)
    {
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_num = pgaspi_segment_num
gaspi_return_t
pgaspi_segment_num (gaspi_number_t * const segment_num)
{
  gaspi_verify_init("gaspi_segment_num");
  gaspi_verify_null_ptr(segment_num);

  *segment_num = (gaspi_number_t) glb_gaspi_ctx.mseg_cnt;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_alloc = pgaspi_segment_alloc
gaspi_return_t
pgaspi_segment_alloc (const gaspi_segment_id_t segment_id,
		      const gaspi_size_t size,
		      const gaspi_alloc_t alloc_policy)
{

  gaspi_verify_init("gaspi_segment_alloc");
  gaspi_verify_segment_size(size);
  gaspi_verify_segment(segment_id);

  if (glb_gaspi_ctx.mseg_cnt >= GASPI_MAX_MSEGS)
    return GASPI_ERR_MANY_SEG;

  lock_gaspi_tout (&gaspi_mseg_lock, GASPI_BLOCK);

  gaspi_return_t eret = GASPI_ERROR;

  /*  TODO: for now like this, but we need to change this */
#ifndef GPI2_CUDA
  long page_size;

  if (glb_gaspi_ctx.rrmd[segment_id] == NULL)
    {
      glb_gaspi_ctx.rrmd[segment_id] = (gaspi_rc_mseg *) calloc (glb_gaspi_ctx.tnc, sizeof (gaspi_rc_mseg));

      if(!glb_gaspi_ctx.rrmd[segment_id])
	{
	  eret = GASPI_ERR_MEMALLOC;
	  goto endL;
	}
    }

  if (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size)
    {
      eret = GASPI_SUCCESS;
      goto endL;
    }

  page_size = sysconf (_SC_PAGESIZE);

  if(page_size < 0)
    {
      gaspi_print_error ("Failed to get system's page size.");
      goto endL;
    }

  if (posix_memalign ((void **) &glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.ptr,
		      page_size,
		      size + NOTIFY_OFFSET) != 0)
    {
      gaspi_print_error ("Memory allocation (posix_memalign) failed");
      eret = GASPI_ERR_MEMALLOC;
      goto endL;
    }

  memset (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.ptr, 0,
	  NOTIFY_OFFSET);

  if (alloc_policy == GASPI_MEM_INITIALIZED)
    memset (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.ptr, 0,
	    size + NOTIFY_OFFSET);

  if(pgaspi_dev_register_mem(&(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank]), size + NOTIFY_OFFSET) < 0)
    {
      goto endL;
    }

  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size = size;

#else
  eret = pgaspi_dev_segment_alloc(segment_id, size, alloc_policy);
  if( eret != GASPI_SUCCESS )
    goto endL;
#endif /* GPI2_CUDA */

  glb_gaspi_ctx.mseg_cnt++;

  eret = GASPI_SUCCESS;
 endL:
  unlock_gaspi (&gaspi_mseg_lock);
  return eret;

}

#pragma weak gaspi_segment_delete = pgaspi_segment_delete
gaspi_return_t
pgaspi_segment_delete (const gaspi_segment_id_t segment_id)
{
  gaspi_verify_init("gaspi_segment_delete");
  gaspi_verify_segment(segment_id);
  gaspi_verify_null_ptr(glb_gaspi_ctx.rrmd[segment_id]);

  gaspi_verify_segment_size(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size);

  gaspi_return_t eret = GASPI_ERROR;

  lock_gaspi_tout(&gaspi_mseg_lock, GASPI_BLOCK);

  /*  TODO: for now like this but we need a better solution */
#ifdef GPI2_CUDA
  eret = pgaspi_dev_segment_delete(segment_id);
#else
  if(pgaspi_dev_unregister_mem(&(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank])) < 0)
    {
      unlock_gaspi (&gaspi_mseg_lock);
      return GASPI_ERR_DEVICE;
    }

  free (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.buf);

  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.buf = NULL;
  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size = 0;
  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].trans = 0;
  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].rkey = 0;

  eret = GASPI_SUCCESS;
#endif

  glb_gaspi_ctx.mseg_cnt--;

  unlock_gaspi (&gaspi_mseg_lock);

  return eret;
}

#pragma weak gaspi_segment_register = pgaspi_segment_register
gaspi_return_t
pgaspi_segment_register(const gaspi_segment_id_t segment_id,
			const gaspi_rank_t rank,
			const gaspi_timeout_t timeout_ms)
{

  gaspi_verify_init("gaspi_segment_register");
  gaspi_verify_segment(segment_id);
  gaspi_verify_null_ptr(glb_gaspi_ctx.rrmd[segment_id]);
  gaspi_verify_rank(rank);

  gaspi_verify_segment_size(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size);

  if( rank == glb_gaspi_ctx.rank)
    return GASPI_SUCCESS;

  if(lock_gaspi_tout(&glb_gaspi_ctx_lock, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  gaspi_return_t eret = gaspi_sn_command(GASPI_SN_SEG_REGISTER, rank, timeout_ms, (void *) &segment_id);

  unlock_gaspi(&glb_gaspi_ctx_lock);

  return eret;
}

static gaspi_return_t
pgaspi_dev_wait_remote_register(const gaspi_segment_id_t segment_id,
				const gaspi_group_t group,
				const gaspi_timeout_t timeout_ms)
{
  int r;
  struct timeb t0, t1;
  ftime(&t0);

  while(1)
    {
      int cnt = 0;

      for(r = 0; r < glb_gaspi_group_ctx[group].tnc; r++)
	{
	  int i = glb_gaspi_group_ctx[group].rank_grp[r];

	  ulong s = gaspi_load_ulong(&glb_gaspi_ctx.rrmd[segment_id][i].size);
	  if(s > 0)
	    cnt++;
	}

      if(cnt == glb_gaspi_group_ctx[group].tnc)
	break;

      ftime(&t1);

      const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);

      if(delta_ms > timeout_ms)
	{
	  return GASPI_TIMEOUT;
	}

      struct timespec sleep_time,rem;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 250000000;
      nanosleep(&sleep_time, &rem);

      //usleep(250000);
    }

  return GASPI_SUCCESS;
}

static gaspi_return_t
pgaspi_segment_register_group(const gaspi_segment_id_t segment_id,
				  const gaspi_group_t group,
				  const gaspi_timeout_t timeout_ms)
{
  int r;
  gaspi_return_t eret = GASPI_ERROR;

  /* register segment to all other group members */
  /* dont write several times !!! */
  for(r = 1; r <= glb_gaspi_group_ctx[group].tnc; r++)
    {
      int i = (glb_gaspi_group_ctx[group].rank + r) % glb_gaspi_group_ctx[group].tnc;

      if(glb_gaspi_group_ctx[group].rank_grp[i] == glb_gaspi_ctx.rank)
	{
	  glb_gaspi_ctx.rrmd[segment_id][i].trans = 1;
	  continue;
	}

      if(glb_gaspi_ctx.rrmd[segment_id][i].trans)
	continue;

      eret = gaspi_sn_command(GASPI_SN_SEG_REGISTER, glb_gaspi_group_ctx[group].rank_grp[i], timeout_ms, (void *) &segment_id);
      if(eret != GASPI_SUCCESS)
	{
	  return eret;
	}

      glb_gaspi_ctx.rrmd[segment_id][i].trans = 1;
    }

  eret = pgaspi_dev_wait_remote_register(segment_id, group, timeout_ms);

  return eret;
}

#pragma weak gaspi_segment_create = pgaspi_segment_create
gaspi_return_t
pgaspi_segment_create(const gaspi_segment_id_t segment_id,
		      const gaspi_size_t size,
		      const gaspi_group_t group,
		      const gaspi_timeout_t timeout_ms,
		      const gaspi_alloc_t alloc_policy)
{
  gaspi_verify_group(group);

  gaspi_return_t eret = pgaspi_segment_alloc (segment_id, size, alloc_policy);

  if(eret != GASPI_SUCCESS)
    {
      return eret;
    }

  if(lock_gaspi_tout(&glb_gaspi_ctx_lock, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_segment_register_group(segment_id, group, timeout_ms);

  unlock_gaspi (&glb_gaspi_ctx_lock);

  eret = pgaspi_barrier(group, timeout_ms);

  return eret;
}
