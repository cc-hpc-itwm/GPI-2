/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2016

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

int
pgaspi_dev_register_mem(gaspi_context_t const * const gctx, gaspi_rc_mseg_t *seg)
{
  gaspi_ib_ctx * const ib_dev_ctx = (gaspi_ib_ctx*) gctx->device->ctx;

  seg->mr[0] = ibv_reg_mr (ib_dev_ctx->pd,
			   seg->data.buf,
			   seg->size,
			   IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
			   IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  if (seg->mr[0] == NULL)
    {
      gaspi_print_error ("Memory registration failed (libibverbs)");
      return -1;
    }

  seg->rkey[0] = ((struct ibv_mr *) seg->mr[0])->rkey;

  if(seg->notif_spc.buf != NULL)
    {
      seg->mr[1] = ibv_reg_mr (ib_dev_ctx->pd,
			       seg->notif_spc.buf,
			       seg->notif_spc_size,
			       IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
			       IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

      if (seg->mr[1] == NULL)
	{
	  gaspi_print_error ("Memory registration failed (libibverbs)");
	  return -1;
	}

      seg->rkey[1] = ((struct ibv_mr *) seg->mr[1])->rkey;
    }

  return 0;
}

int
pgaspi_dev_unregister_mem(gaspi_context_t const * const gctx,const gaspi_rc_mseg_t * seg)
{
  if( seg->mr[0] != NULL)
    {

      if (ibv_dereg_mr ((struct ibv_mr *)seg->mr[0]))
	{
	  gaspi_print_error ("Memory de-registration failed (libibverbs)");
	  return -1;
	}
    }

  if( seg->mr[1] != NULL)
    {
      if (ibv_dereg_mr ((struct ibv_mr *)seg->mr[1]))
	{
	  gaspi_print_error ("Memory de-registration failed (libibverbs)");
	  return -1;
	}
    }

  return 0;
}

#ifdef GPI2_CUDA
gaspi_return_t
pgaspi_dev_segment_alloc (const gaspi_segment_id_t segment_id,
			  const gaspi_size_t size,
			  const gaspi_alloc_t alloc_policy)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if (gctx->rrmd[segment_id] == NULL)
    {
      gctx->rrmd[segment_id] = (gaspi_rc_mseg_t *) calloc (gctx->tnc, sizeof (gaspi_rc_mseg_t));

      if(!gctx->rrmd[segment_id])
	goto errL;
    }

  if (gctx->rrmd[segment_id][gctx->rank].size)
    {
      goto okL;
    }

  if( alloc_policy & GASPI_MEM_GPU )
    {
      if( size > GASPI_GPU_MAX_SEG )
	{
	  gaspi_print_error("Segment size too large for GPU Segment (max %u).", GASPI_GPU_MAX_SEG);
	  goto errL;
	}

      cudaError_t cuda_error_id = cudaGetDevice(&gctx->rrmd[segment_id][gctx->rank].cuda_dev_id);
      if( cuda_error_id != cudaSuccess )
	{
	  gaspi_print_error("Failed cudaGetDevice." );
	  return GASPI_ERROR;
	}

      gaspi_gpu_t* agpu =  _gaspi_find_gpu(gctx->rrmd[segment_id][gctx->rank].cuda_dev_id);
      if( !agpu )
	{
	  gaspi_print_error("No GPU found. Maybe forgot to call gaspi_init_GPUs?");
	  goto errL;
	}

      /* Allocate device memory for data */
      if( cudaMalloc((void**)&gctx->rrmd[segment_id][gctx->rank].data.ptr, size ) != 0)
	{
	  gaspi_print_error("GPU memory allocation (cudaMalloc) failed.");
	  goto errL;
	}

      /* Allocate host memory for data and notifications */
      if( cudaMallocHost((void**)&gctx->rrmd[segment_id][gctx->rank].host_ptr, size + NOTIFY_OFFSET ) != 0)
	{
	  gaspi_print_error("Memory allocattion (cudaMallocHost)  failed.");
	  goto errL;
	}

      memset(gctx->rrmd[segment_id][gctx->rank].host_ptr, 0, size + NOTIFY_OFFSET);

      /* Register host memory */
      gctx->rrmd[segment_id][gctx->rank].host_mr =
	ibv_reg_mr( glb_gaspi_ctx_ib.pd,
		    gctx->rrmd[segment_id][gctx->rank].host_ptr,
		    NOTIFY_OFFSET + size,
		    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		    IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_ATOMIC);

      if( gctx->rrmd[segment_id][gctx->rank].host_mr == NULL )
	{
	  gaspi_print_error("Memory registration failed (libibverbs).");
	  goto errL;
	}

      if( alloc_policy == GASPI_MEM_INITIALIZED )
	{
	  cudaMemset(gctx->rrmd[segment_id][gctx->rank].data.ptr, 0, size);
	}

      /* Register device memory */
      gctx->rrmd[segment_id][gctx->rank].mr[0] =
	ibv_reg_mr( glb_gaspi_ctx_ib.pd,
		    gctx->rrmd[segment_id][gctx->rank].data.buf,
		    size,
		    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

      if( gctx->rrmd[segment_id][gctx->rank].mr[0] == NULL )
	{
	  gaspi_print_error ("Memory registration failed (libibverbs)");
	  goto errL;
	}

      gctx->rrmd[segment_id][gctx->rank].host_rkey =
	((struct ibv_mr *) gctx->rrmd[segment_id][gctx->rank].host_mr)->rkey;

      gctx->rrmd[segment_id][gctx->rank].host_addr = (uintptr_t)gctx->rrmd[segment_id][gctx->rank].host_ptr;
    }
  else
    {
      gctx->rrmd[segment_id][gctx->rank].cuda_dev_id = -1;
      gctx->rrmd[segment_id][gctx->rank].host_rkey = 0;
      gctx->rrmd[segment_id][gctx->rank].host_addr = 0;
      if( gctx->use_gpus != 0 && gctx->gpu_count == 0 )
	{
	  if( cudaMallocHost((void**)&gctx->rrmd[segment_id][gctx->rank].data.ptr, size + NOTIFY_OFFSET))
	    {
	      gaspi_print_error("Memory allocation (cudaMallocHost) failed.");
	      goto errL;
	    }
	}

      memset( gctx->rrmd[segment_id][gctx->rank].data.ptr, 0,
	      NOTIFY_OFFSET);

      if( alloc_policy == GASPI_MEM_INITIALIZED )
	{
	  memset (gctx->rrmd[segment_id][gctx->rank].data.ptr,
		  0,
		  size + NOTIFY_OFFSET);
	}

      gctx->rrmd[segment_id][gctx->rank].size = size;
      gctx->rrmd[segment_id][gctx->rank].notif_spc_size = NOTIFY_OFFSET;
      gctx->rrmd[segment_id][gctx->rank].notif_spc.addr = gctx->rrmd[segment_id][gctx->rank].data.addr;
      gctx->rrmd[segment_id][gctx->rank].data.addr += NOTIFY_OFFSET;
      gctx->rrmd[segment_id][gctx->rank].user_provided = 0;

      if( pgaspi_dev_register_mem(&(gctx->rrmd[segment_id][gctx->rank]) ) < 0)
	{
	  goto errL;
	}
    }

 okL:
  return GASPI_SUCCESS;

 errL:
  return GASPI_ERROR;
}

gaspi_return_t
pgaspi_dev_segment_delete (const gaspi_segment_id_t segment_id)
{
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( gctx->use_gpus != 0 && gctx->gpu_count > 0 )
    {
      if( ibv_dereg_mr (gctx->rrmd[segment_id][gctx->rank].mr[0]) )
	{
	  gaspi_print_error ("Memory de-registration failed (libibverbs)");
	  goto errL;
	}
    }

  if( gctx->rrmd[segment_id][gctx->rank].cuda_dev_id >= 0 )
    {
      if( ibv_dereg_mr (gctx->rrmd[segment_id][gctx->rank].host_mr) )
	{
	  gaspi_print_error ("Memory de-registration failed (libibverbs)");
	  goto errL;
	}

      cudaFreeHost(gctx->rrmd[segment_id][gctx->rank].host_ptr);
      gctx->rrmd[segment_id][gctx->rank].host_ptr = NULL;
      cudaFree(gctx->rrmd[segment_id][gctx->rank].data.buf);
    }
  else if( gctx->use_gpus != 0 && gctx->gpu_count > 0 )
    {
      cudaFreeHost(gctx->rrmd[segment_id][gctx->rank].data.buf);
    }

  free (gctx->rrmd[segment_id][gctx->rank].data.buf);
  gctx->rrmd[segment_id][gctx->rank].data.buf = NULL;

  memset(gctx->rrmd[segment_id], 0, gctx->tnc * sizeof (gaspi_rc_mseg_t));
  free(gctx->rrmd[segment_id]);
  gctx->rrmd[segment_id] = NULL;

  return GASPI_SUCCESS;

 errL:
  return GASPI_ERROR;
}
#endif // GPI2_CUDA
