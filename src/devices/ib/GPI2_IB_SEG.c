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

int
pgaspi_dev_register_mem(const gaspi_segment_id_t segment_id, const gaspi_size_t size)
{
  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr =
    ibv_reg_mr (glb_gaspi_ctx_ib.pd,
		glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf,
		size + NOTIFY_OFFSET,
		IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  if (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr == NULL) 
    {
      gaspi_print_error ("Memory registration failed (libibverbs)");
      return -1;
    }

  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].rkey =
    ((struct ibv_mr *) glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr)->rkey;


  return 0;
}

int
pgaspi_dev_unregister_mem(const gaspi_segment_id_t segment_id)
{
 
  if (ibv_dereg_mr ((struct ibv_mr *)glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr))
    {
      gaspi_print_error ("Memory de-registration failed (libibverbs)");
      return -1;
    }
  
  return 0;
}

#ifdef GPI2_CUDA
gaspi_return_t
pgaspi_dev_segment_alloc (const gaspi_segment_id_t segment_id,
			  const gaspi_size_t size,
			  const gaspi_alloc_t alloc_policy)
{
  unsigned int page_size;
  
  if (glb_gaspi_ctx.rrmd[segment_id] == NULL)
    {
      glb_gaspi_ctx.rrmd[segment_id] = (gaspi_rc_mseg *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));

      if(!glb_gaspi_ctx.rrmd[segment_id])
	goto errL;

      memset (glb_gaspi_ctx.rrmd[segment_id], 0,glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
    }

  if (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size)
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
      
      cudaGetDevice(&glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId);
      gaspi_gpu* agpu =  _gaspi_find_gpu(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId);
      if(!agpu)
	{
	  gaspi_print_error("No GPU found. Maybe forgot to call gaspi_init_GPUs?\n");
	  goto errL;
	}
      
      if(cudaMalloc((void**)&glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr,size) != 0)
	{
	  gaspi_print_error("GPU memory allocation (cudaMalloc) failed!\n");
	  goto errL;
	}
      
      if(cudaMallocHost((void**)&glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr,size+NOTIFY_OFFSET)!=0)
	{
	  gaspi_print_error("Memory allocattion (cudaMallocHost)  failed!\n");
	  goto errL;
	}

      memset(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr, 0, size+NOTIFY_OFFSET);

      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_mr =
	ibv_reg_mr(glb_gaspi_ctx_ib.pd,glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr,
		   NOTIFY_OFFSET+size,IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_ATOMIC);

      if(!glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_mr)
	{
	  gaspi_print_error("Memory registration failed (libibverbs)\n");
	  goto errL;
	}
      
      if(alloc_policy == GASPI_MEM_INITIALIZED)
	cudaMemset(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr,0,size);

      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr =
	ibv_reg_mr (glb_gaspi_ctx_ib.pd,
		    glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf,
		    size,
		    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

      if (!glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr)
	{
	  gaspi_print_error ("Memory registration failed (libibverbs)");
	  goto errL;
	}
      
      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_rkey =
	((struct ibv_mr *) glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_mr)->rkey;
      
      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_addr = (uintptr_t)glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr;
    }
  else
    {
      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId = -1;
      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_rkey = 0;
      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_addr = 0;
      if(glb_gaspi_ctx.use_gpus!=0 &&glb_gaspi_ctx.gpu_count==0)
	{
	  if( cudaMallocHost((void**)&glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, size+NOTIFY_OFFSET))
	    {
	      gaspi_print_error("Memory allocation (cudaMallocHost) failed !\n");
	      goto errL;
	    }
	}
      else
#endif
	if (posix_memalign
	    ((void **) &glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr,
	     page_size, size + NOTIFY_OFFSET) != 0)
	  {
	    gaspi_print_error ("Memory allocation (posix_memalign) failed");
	    goto errL;
	  }
      
      memset (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, 0,
	      NOTIFY_OFFSET);

      if (alloc_policy == GASPI_MEM_INITIALIZED)
	memset (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, 0,
		size + NOTIFY_OFFSET);

#ifdef GPI2_CUDA
      if(glb_gaspi_ctx.use_gpus == 0 || glb_gaspi_ctx.gpu_count == 0)
#endif
	glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr =
	  ibv_reg_mr (glb_gaspi_ctx_ib.pd,
		      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf,
		      size + NOTIFY_OFFSET,
		      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  
      if (!glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr)
	{
	  gaspi_print_error ("Memory registration failed (libibverbs)");
	  goto errL;
	}
#ifdef GPI2_CUDA
    }
#endif

  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].rkey =
    ((struct ibv_mr *) glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr)->rkey;
  
  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].addr =
    (uintptr_t) glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf;

  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size = size;

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
  
    if (ibv_dereg_mr (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].mr))
      {
	gaspi_print_error ("Memory de-registration failed (libibverbs)");
	goto errL;
      }

#ifdef GPI2_CUDA
  if(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].cudaDevId >= 0)
    {
      if (ibv_dereg_mr (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_mr))
	{
	  gaspi_print_error ("Memory de-registration failed (libibverbs)");
	  goto errL;
	}
      
      cudaFreeHost(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr);
      glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_ptr = NULL;
      cudaFree(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf);
    }
  else if(glb_gaspi_ctx.use_gpus != 0 && glb_gaspi_ctx.gpu_count > 0)
    {
      cudaFreeHost(glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf);
    }
  
  else
#endif
    free (glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf);

  glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].buf = NULL;

  memset(glb_gaspi_ctx.rrmd[segment_id], 0, glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));

  free(glb_gaspi_ctx.rrmd[segment_id]);

  glb_gaspi_ctx.rrmd[segment_id] = NULL;

  return GASPI_SUCCESS;

 errL:
  return GASPI_ERROR;
}
#endif // GPI2_CUDA

