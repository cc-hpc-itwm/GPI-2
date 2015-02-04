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

#include <errno.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <unistd.h>
#ifdef GPI2_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "GASPI_GPU.h"
#include "GPI2_GPU.h"
#endif

#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_IB.h"
#include "GPI2_SN.h"

inline void
pgaspi_dev_set_mseg_trans(const gaspi_segment_id_t segment_id, const gaspi_rank_t rank, const int val)
{
  glb_gaspi_ctx_ib.rrmd[segment_id][rank].trans = val;
}


inline int
pgaspi_dev_get_mseg_trans(const gaspi_segment_id_t segment_id, const gaspi_rank_t rank)
{
  return glb_gaspi_ctx_ib.rrmd[segment_id][rank].trans;
}

gaspi_return_t
pgaspi_dev_segment_alloc (const gaspi_segment_id_t segment_id,
			  const gaspi_size_t size,
			  const gaspi_alloc_t alloc_policy)
{
  unsigned int page_size;
  
  if (glb_gaspi_ctx_ib.rrmd[segment_id] == NULL)
    {
      glb_gaspi_ctx_ib.rrmd[segment_id] = (gaspi_rc_mseg *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));

      if(!glb_gaspi_ctx_ib.rrmd[segment_id])
	goto errL;

      memset (glb_gaspi_ctx_ib.rrmd[segment_id], 0,glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
    }

  if (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size)
    {
      goto okL;
    }

  page_size = sysconf (_SC_PAGESIZE);
  //TODO: error check on page_size
  
#ifdef GPI2_CUDA
  if(alloc_policy&GASPI_MEM_GPU)
    {
      if(size > GASPI_GPU_MAX_SEG)
	{
	  gaspi_print_error("Segment size too large for GPU Segment (max %u)\n", GASPI_GPU_MAX_SEG);
	  goto errL;
	}
      
      cudaGetDevice(&glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId);
      gaspi_gpu* agpu =  _gaspi_find_gpu(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId);
      if(!agpu)
	{
	  gaspi_print_error("No GPU found. Maybe forgot to call gaspi_init_GPUs?\n");
	  goto errL;
	}
      
      if(cudaMalloc((void**)&glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr,size) != 0)
	{
	  gaspi_print_error("GPU memory allocation (cudaMalloc) failed!\n");
	  goto errL;
	}
      
      if(cudaMallocHost((void**)&glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr,size+NOTIFY_OFFSET)!=0)
	{
	  gaspi_print_error("Memory allocattion (cudaMallocHost)  failed!\n");
	  goto errL;
	}

      memset(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr, 0, size+NOTIFY_OFFSET);

      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_mr =
	ibv_reg_mr(glb_gaspi_ctx_ib.pd,glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr,
		   NOTIFY_OFFSET+size,IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_ATOMIC);

      if(!glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_mr)
	{
	  gaspi_print_error("Memory registration failed (libibverbs)\n");
	  goto errL;
	}
      
      if(alloc_policy == GASPI_MEM_INITIALIZED)
	cudaMemset(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr,0,size);

      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr =
	ibv_reg_mr (glb_gaspi_ctx_ib.pd,
		    glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf,
		    size,
		    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

      if (!glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr)
	{
	  gaspi_print_error ("Memory registration failed (libibverbs)");
	  goto errL;
	}
      
      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_rkey = glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_mr->rkey;
      
      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_addr = (uintptr_t)glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr;
    }
  else
    {
      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId = -1;
      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_rkey = 0;
      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_addr = 0;
      if(glb_gaspi_ctx.use_gpus!=0 &&glb_gaspi_ctx.gpu_count==0)
	{
	  if( cudaMallocHost((void**)&glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, size+NOTIFY_OFFSET))
	    {
	      gaspi_print_error("Memory allocation (cudaMallocHost) failed !\n");
	      goto errL;
	    }
	}
      else
#endif
	if (posix_memalign
	    ((void **) &glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr,
	     page_size, size + NOTIFY_OFFSET) != 0)
	  {
	    gaspi_print_error ("Memory allocation (posix_memalign) failed");
	    goto errL;
	  }
      
      memset (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, 0,
	      NOTIFY_OFFSET);

      if (alloc_policy == GASPI_MEM_INITIALIZED)
	memset (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, 0,
		size + NOTIFY_OFFSET);

#ifdef GPI2_CUDA
      if(glb_gaspi_ctx.use_gpus == 0 || glb_gaspi_ctx.gpu_count == 0)
#endif
	glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr =
	  ibv_reg_mr (glb_gaspi_ctx_ib.pd,
		      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf,
		      size + NOTIFY_OFFSET,
		      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  
      if (!glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr)
	{
	  gaspi_print_error ("Memory registration failed (libibverbs)");
	  goto errL;
	}
#ifdef GPI2_CUDA
    }
#endif

  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].rkey =
    glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr->rkey;
  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].addr =
    (uintptr_t) glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf;

  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size = size;

 okL:
  return GASPI_SUCCESS;

 errL:
  return GASPI_ERROR;

}

gaspi_return_t
pgaspi_dev_segment_delete (const gaspi_segment_id_t segment_id)
{
  
#ifdef GPI2_CUDA 
  if(glb_gaspi_ctx.use_gpus != 0 && glb_gaspi_ctx.gpu_count > 0)
#endif
  
    //potential problem: different threads are allocating/registering. should not be a problem ?
    if (ibv_dereg_mr (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr))
      {
	gaspi_print_error ("Memory de-registration failed (libibverbs)");
	goto errL;
      }

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId >= 0)
    {
      if (ibv_dereg_mr (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_mr))
	{
	  gaspi_print_error ("Memory de-registration failed (libibverbs)");
	  goto errL;
	}
      
      cudaFreeHost(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr);
      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr = NULL;
      cudaFree(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf);
    }
  else if(glb_gaspi_ctx.use_gpus != 0 && glb_gaspi_ctx.gpu_count > 0)
    {
      cudaFreeHost(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf);
    }
  
  else
#endif
    free (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf);

  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf = NULL;

  memset(glb_gaspi_ctx_ib.rrmd[segment_id], 0, glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));

  free(glb_gaspi_ctx_ib.rrmd[segment_id]);

  glb_gaspi_ctx_ib.rrmd[segment_id] = NULL;

  return GASPI_SUCCESS;

 errL:
  return GASPI_ERROR;
}


void
pgaspi_dev_init_cdh_segment(const gaspi_segment_id_t segment_id,
			    const gaspi_rank_t rank,
			    gaspi_cd_header *cdh)
{
  
  cdh->op_len = 0;// in place
  cdh->op = GASPI_SN_SEG_REGISTER;
  cdh->rank = rank;
  cdh->seg_id = segment_id;
  cdh->rkey = glb_gaspi_ctx_ib.rrmd[segment_id][rank].rkey;
  cdh->addr = glb_gaspi_ctx_ib.rrmd[segment_id][rank].addr;
  cdh->size = glb_gaspi_ctx_ib.rrmd[segment_id][rank].size;

#ifdef GPI2_CUDA
  cdh->host_rkey = glb_gaspi_ctx_ib.rrmd[segment_id][rank].host_rkey;
  cdh->host_addr = glb_gaspi_ctx_ib.rrmd[segment_id][rank].host_addr;
#endif
}

//sn-registration
//TODO: timeout?
int
gaspi_seg_reg_sn(const gaspi_cd_header snp)
{

  if(!glb_gaspi_ib_init) 
    return GASPI_ERROR;

  //TODO: include timeout?
  lock_gaspi_tout(&gaspi_mseg_lock,GASPI_BLOCK);

  if(glb_gaspi_ctx_ib.rrmd[snp.seg_id] == NULL)
    {
      glb_gaspi_ctx_ib.rrmd[snp.seg_id] = (gaspi_rc_mseg *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));

      if(!glb_gaspi_ctx_ib.rrmd[snp.seg_id]) 
	goto errL;
      
    memset(glb_gaspi_ctx_ib.rrmd[snp.seg_id], 0, glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
  }

  //TODO: don't allow re-registration
  //for now we allow re-registration
  //if(glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rem_rank].size) -> re-registration error case

  glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].rkey = snp.rkey;
  glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].addr = snp.addr;
  glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].size = snp.size;

#ifdef GPI2_CUDA
  glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].host_rkey = snp.host_rkey;
  glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].host_addr = snp.host_addr;

  if(snp.host_addr != 0)
    glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].cudaDevId = 1;
  else
    glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].cudaDevId = -1;
#endif

  unlock_gaspi(&gaspi_mseg_lock);
  return 0;

errL:
  unlock_gaspi(&gaspi_mseg_lock);
  return -1;

}

gaspi_return_t
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
	  ulong s = gaspi_load_ulong(&glb_gaspi_ctx_ib.rrmd[segment_id][i].size);
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



gaspi_return_t
pgaspi_dev_segment_ptr (const gaspi_segment_id_t segment_id, gaspi_pointer_t * ptr)
{

#ifdef GPI2_CUDA
  
  if(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId >= 0)
    *ptr = glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf;
  else
    
#endif
    
    *ptr = glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf + NOTIFY_OFFSET;
  
  return GASPI_SUCCESS;
}
