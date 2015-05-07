/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2014

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
  gaspi_size_t seg_size = glb_gaspi_ctx.rrmd[segment_id][rank].size;
  
#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if (glb_gaspi_ctx.rrmd[segment_id] == NULL)
    {
      gaspi_print_error("Invalid segment id");
      return GASPI_ERROR;
    }

  if (0 == seg_size)
    {
      gaspi_print_error("Invalid segment (size = 0)");
      return GASPI_ERROR;
    }

  gaspi_verify_null_ptr(size);

#endif
  
  *size = seg_size;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_ptr = pgaspi_segment_ptr
gaspi_return_t
pgaspi_segment_ptr (const gaspi_segment_id_t segment_id, gaspi_pointer_t * ptr)
{

#ifdef DEBUG  
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if (glb_gaspi_ctx.rrmd[segment_id] == NULL)
    {
      gaspi_print_error("Invalid segment id");
      return GASPI_ERROR;
    }

  if (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size == 0)
    {
      gaspi_print_error("Invalid segment (size = 0)");
      return GASPI_ERROR;
    }

  gaspi_verify_null_ptr(ptr);
#endif

#ifdef GPI2_CUDA
  
  if(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId >= 0)
    *ptr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf;
  else
    
#endif
    
    *ptr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf + NOTIFY_OFFSET;
  
  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_list = pgaspi_segment_list
gaspi_return_t
pgaspi_segment_list (const gaspi_number_t num,
		     gaspi_segment_id_t * const segment_id_list)
{
  int i, idx = 0;

#ifdef DEBUG    
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
#endif
  
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
  if (glb_gaspi_init)
    {
      *segment_num = (gaspi_number_t) glb_gaspi_ctx.mseg_cnt;
      return GASPI_SUCCESS;
    }

  gaspi_print_error("Invalid function before gaspi_proc_init");
  return GASPI_ERROR;
}

#pragma weak gaspi_segment_alloc = pgaspi_segment_alloc
gaspi_return_t
pgaspi_segment_alloc (const gaspi_segment_id_t segment_id,
		      const gaspi_size_t size,
		      const gaspi_alloc_t alloc_policy)
{

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if (glb_gaspi_ctx.mseg_cnt >= GASPI_MAX_MSEGS || size == 0)
    return GASPI_ERROR;

  lock_gaspi_tout (&gaspi_mseg_lock, GASPI_BLOCK);

  gaspi_return_t eret = GASPI_ERROR;

  /*  TODO: for now like this, but we need to change this */
#ifndef GPI2_CUDA  
  unsigned int page_size;
    
  if (glb_gaspi_ctx.rrmd[segment_id] == NULL)
    {
      glb_gaspi_ctx.rrmd[segment_id] = (gaspi_rc_mseg *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
      
      if(!glb_gaspi_ctx.rrmd[segment_id])
	  goto endL;

      memset (glb_gaspi_ctx.rrmd[segment_id], 0,glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
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

  if (posix_memalign ((void **) &glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr,
		      page_size,
		      size + NOTIFY_OFFSET) != 0)
    {
      gaspi_print_error ("Memory allocation (posix_memalign) failed");
      goto endL;
    }
  
  memset (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, 0,
	  NOTIFY_OFFSET);
  
  if (alloc_policy == GASPI_MEM_INITIALIZED)
    memset (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, 0,
	    size + NOTIFY_OFFSET);

  if(pgaspi_dev_register_mem(&(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank]), size + NOTIFY_OFFSET) < 0)
    {
      goto endL;
    }
  
  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].addr =
    (unsigned long) glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf;

  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size = size;
    
#else
  eret = pgaspi_dev_segment_alloc(segment_id, size, alloc_policy);
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

  if(!glb_gaspi_dev_init)
    {
      return GASPI_ERROR;
    }

#ifdef DEBUG  
  if (glb_gaspi_ctx.rrmd[segment_id] == NULL)
    {
      gaspi_print_error("Invalid segment to delete");
      return GASPI_ERROR;
    }
  
  if (0 == glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size)
    {
      gaspi_print_error("Invalid segment to delete");
      return GASPI_ERROR;
    }
  
#endif
  gaspi_return_t eret = GASPI_ERROR;
  
  lock_gaspi_tout(&gaspi_mseg_lock,GASPI_BLOCK);

  /*  TODO: for now like this but we need a better solution */
#ifdef GPI2_CUDA  
  eret = pgaspi_dev_segment_delete(segment_id);
#else
  if(pgaspi_dev_unregister_mem(&(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank])) < 0)
    {
      unlock_gaspi (&gaspi_mseg_lock);
      return GASPI_ERROR;
    }
  
  free (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf);

  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf = NULL;

  memset(glb_gaspi_ctx.rrmd[segment_id], 0, glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));

  free(glb_gaspi_ctx.rrmd[segment_id]);

  glb_gaspi_ctx.rrmd[segment_id] = NULL;

  eret = GASPI_SUCCESS;
#endif
  
  glb_gaspi_ctx.mseg_cnt--;
  
  unlock_gaspi (&gaspi_mseg_lock);

  return eret;
}

static int
_gaspi_dev_exch_cdh(gaspi_cd_header *cdh,
		    gaspi_rank_t rank)
{
  int ret = write(glb_gaspi_ctx.sockfd[rank], cdh, sizeof(gaspi_cd_header));
  if(ret != sizeof(gaspi_cd_header))
    {
      gaspi_print_error("Failed to write (%d %p %lu)",
			glb_gaspi_ctx.sockfd[rank], cdh, sizeof(gaspi_cd_header));

      return -1;
    }

  int rret;
  ret = read(glb_gaspi_ctx.sockfd[rank], &rret, sizeof(int));
  if(ret != sizeof(int))
    {
      gaspi_print_error("Failed to read from rank %d", rank);
      return -1;
    }
  
  if(rret < 0)
    {
      gaspi_print_error("Read (unexpectedly) 0 bytes from %d\n", rank);
      return -1;
    }

  return 0;
}

gaspi_return_t
_pgaspi_segment_register(const gaspi_segment_id_t segment_id,
			 const gaspi_rank_t rank,
			 const gaspi_timeout_t timeout_ms)
  
{
  gaspi_return_t eret = gaspi_connect_to_rank(rank, timeout_ms);
  if(eret != GASPI_SUCCESS)
    {
      return eret;
    }

  gaspi_cd_header cdh;

  cdh.op_len = 0;// in place
  cdh.op = GASPI_SN_SEG_REGISTER;
  cdh.rank = glb_gaspi_ctx.rank;
  cdh.seg_id = segment_id;
  cdh.rkey = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].rkey;
  cdh.addr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].addr;
  cdh.size = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size;

#ifdef GPI2_CUDA
  cdh.host_rkey = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_rkey;
  cdh.host_addr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_addr;
#endif

  if(_gaspi_dev_exch_cdh(&cdh, rank) != 0)
    {
      gaspi_print_error("Failed to exch cdh with %d", rank);
      return GASPI_ERROR;
    }
    
  if(gaspi_close(glb_gaspi_ctx.sockfd[rank]) != 0)
    {
      gaspi_print_error("Failed to close connection to %d", rank);

      glb_gaspi_ctx.qp_state_vec[GASPI_SN][rank] = 1;
      
      return GASPI_ERROR;
    }

  glb_gaspi_ctx.sockfd[rank] = -1;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_register = pgaspi_segment_register
gaspi_return_t
pgaspi_segment_register(const gaspi_segment_id_t segment_id,
			const gaspi_rank_t rank,
			const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG  
  if(!glb_gaspi_dev_init)
    return GASPI_ERROR;

  if(rank >= glb_gaspi_ctx.tnc || rank == glb_gaspi_ctx.rank)
    return GASPI_ERROR;

  if(glb_gaspi_ctx.rrmd[segment_id] == NULL)
    return GASPI_ERROR;
  
  if(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size == 0)
    return GASPI_ERROR;
#endif

  gaspi_return_t eret = GASPI_ERROR;

  if(lock_gaspi_tout(&glb_gaspi_ctx_lock, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  eret = _pgaspi_segment_register(segment_id, rank, timeout_ms);

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

	  //TODO: only here is ctx_ib needed
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

      eret = _pgaspi_segment_register(segment_id, glb_gaspi_group_ctx[group].rank_grp[i], timeout_ms);
      if(eret != GASPI_SUCCESS)
	{
	  gaspi_print_error("Failed segment registration with %d\n",
			    glb_gaspi_group_ctx[group].rank_grp[i]);
	  
	  return GASPI_ERROR;
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
#ifdef DEBUG  
  if(!glb_gaspi_dev_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if(group >= GASPI_MAX_GROUPS || glb_gaspi_group_ctx[group].id < 0)
    {
      gaspi_print_error("Invalid group ( > GASPI_MAX_GROUPS || < 0)");
      return GASPI_ERROR;
    }
#endif

  gaspi_return_t eret = GASPI_ERROR;
  
  if(pgaspi_segment_alloc (segment_id, size, alloc_policy) != GASPI_SUCCESS)
    {
      gaspi_print_error("Segment allocation failed");
      eret = GASPI_ERROR;
      goto endL;
    }

  if(lock_gaspi_tout(&glb_gaspi_ctx_lock, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  eret = pgaspi_segment_register_group(segment_id, group, timeout_ms);

 endL:
  
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}    
