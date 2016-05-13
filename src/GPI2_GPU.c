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

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string.h>

#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_GPU.h"
#include "GASPI_GPU.h"

#define GPI2_GPU_MAX_DIRECT_DEVS 32

/* Look for a GPU with a particular device identifier */
gaspi_gpu_t*
_gaspi_find_gpu(int dev_id)
{
  int i;
  for (i = 0; i < glb_gaspi_ctx.gpu_count; i++)
    if( gpus[i].device_id == dev_id )
      {
	cudaSetDevice(dev_id);
	return &gpus[i];
      }

  return NULL;
}

static int
_gaspi_find_GPU_numa_node(int cudevice)
{
  CUresult cres;
  int domain, bus, dev;
  char path[128];
  FILE *sysfile = NULL;

  domain = 0;

#ifdef CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
  cres = cuDeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, cudevice);
  if( CUDA_SUCCESS != cres )
    {
      errno = ENOSYS;
      return -1;
    }
#endif

  cres = cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cudevice);
  if( CUDA_SUCCESS != cres )
    {
      return -1;
    }

  cres = cuDeviceGetAttribute(&dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cudevice);
  if( CUDA_SUCCESS != cres )
    {
      return -1;
    }

  sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node", domain, bus, dev);
  sysfile = fopen(path, "r");
  if( !sysfile )
    {
      gaspi_print_error("Failed to open %s.", path);
      return -1;
    }

  int numa_node = -1;
  fscanf (sysfile, "%1d", &numa_node);
  fclose(sysfile);

  return numa_node;
}

gaspi_return_t
gaspi_gpu_init(void)
{
  gaspi_context const * const gctx = &glb_gaspi_ctx;
  int deviceCount;
  cudaError_t cuda_error_id = cudaGetDeviceCount(&deviceCount);
  if( cuda_error_id != cudaSuccess )
    {
      gaspi_print_error("Failed cudaGetDeviceCount." );
      return GASPI_ERR_DEVICE;
    }

  if( deviceCount <= 0 )
    {
      gaspi_print_error("No CUDA capable devices found.");
      return GASPI_ERR_DEVICE;
    }

  const int ib_numa_node = _gaspi_find_dev_numa_node();

  int device_id = 0;
  int gaspi_devices = 0;
  int direct_devices[GPI2_GPU_MAX_DIRECT_DEVS];
  struct cudaDeviceProp deviceProp;
  for(device_id = 0; device_id < deviceCount; device_id++)
    {
      //TODO: possibly add functionality to show properties structure
      cuda_error_id = cudaGetDeviceProperties(&deviceProp, device_id);
      if( cuda_error_id != cudaSuccess)
	{
	  return GASPI_ERR_DEVICE;
	}

      if( deviceProp.major >= 3 ) /* TODO: magic number */
	{
	  cuda_error_id = cudaSetDevice(device_id);
	  if( cuda_error_id != cudaSuccess )
	    {
	      return GASPI_ERR_DEVICE;
	    }

	  if( ib_numa_node == _gaspi_find_GPU_numa_node(device_id) )
	    {
	      if( gaspi_devices < GPI2_GPU_MAX_DIRECT_DEVS - 1 )
		{
		  direct_devices[gaspi_devices] = device_id;
		  gaspi_devices++;
		}
	    }
	}
    }

  if( 0 == gaspi_devices )
    {
      gaspi_print_error("No GPU Direct RDMA capable devices on the correct NUMA-socket were found.");
      return GASPI_ERROR;
    }

  gpus = (gaspi_gpu_t*) malloc(sizeof(gaspi_gpu_t) * gaspi_devices);
  if( gpus == NULL )
    {
      gaspi_print_error("Failed to allocate memory.");
      return GASPI_ERR_MEMALLOC;
    }

  int i, j, k;
  for(k = 0 ; k < gaspi_devices; k++)
    {
      cuda_error_id = cudaSetDevice(direct_devices[k]);
      if( cuda_error_id != cudaSuccess )
	{
	  return GASPI_ERR_DEVICE;
	}

      for(i = 0; i < GASPI_MAX_QP; i++)
	{
	  cuda_error_id = cudaStreamCreate(&gpus[k].streams[i]);
	  if( cuda_error_id != cudaSuccess )
	    {
	      return GASPI_ERR_DEVICE;
	    }

	  for(j = 0; j < GASPI_CUDA_EVENTS; j++)
	    {
	      cuda_error_id = cudaEventCreateWithFlags(&gpus[k].events[i][j].event, cudaEventDisableTiming);
	      if( cuda_error_id != cudaSuccess )
		{
		  return GASPI_ERR_DEVICE;
		}
	    }

	  cuda_error_id = cudaStreamCreateWithFlags(&gpus[k].streams[i], cudaStreamNonBlocking);
	  if( cuda_error_id != cudaSuccess )
	    {
	      return GASPI_ERR_DEVICE;
	    }

	}

      gpus[k].device_id = direct_devices[k];
    }

  gctx->gpu_count = gaspi_devices;
  gctx->use_gpus = 1;

  return GASPI_SUCCESS;
}

gaspi_return_t
gaspi_gpu_number(gaspi_number_t* num_gpus)
{
  gaspi_verify_init("gaspi_gpu_number");
  gaspi_verify_null_ptr(num_gpus);
  gaspi_context const * const gctx = &glb_gaspi_ctx;

  if( 0 == gctx->use_gpus )
    {
      gaspi_print_error("GPUs are not initialized.");
      return GASPI_ERROR;
    }

  *num_gpus = gctx->gpu_count;

  return GASPI_SUCCESS;
}

/* TODO: Do we really need this function or at least make it part of
   the GPU interface and allow clients to use it? */
gaspi_return_t
gaspi_gpu_ids(gaspi_gpu_id_t* gpu_ids)
{
  gaspi_verify_init("gaspi_gpu_ids");
  gaspi_verify_null_ptr(gpu_ids);

  gaspi_context const * const gctx = &glb_gaspi_ctx;

  if( 0 == gctx->use_gpus )
    {
      gaspi_print_error("GPUs are not found/initialized.");
      return GASPI_ERROR;
    }

  int i;
  for(i = 0; i < gctx->gpu_count; i++)
    {
      gpu_ids[i] = gpus[i].device_id;
    }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_gpu_write = pgaspi_gpu_write
gaspi_return_t
pgaspi_gpu_write(const gaspi_segment_id_t segment_id_local,
		 const gaspi_offset_t offset_local,
		 const gaspi_rank_t rank,
		 const gaspi_segment_id_t segment_id_remote,
		 const gaspi_offset_t offset_remote,
		 const gaspi_size_t size,
		 const gaspi_queue_id_t queue,
		 const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_gpu_write");
  gaspi_verify_local_off(offset_local, segment_id_local, size);
  gaspi_verify_remote_off(offset_remote, segment_id_remote, rank, size);
  gaspi_verify_queue(queue);
  gaspi_verify_comm_size(size, segment_id_local, segment_id_remote, rank, GASPI_MAX_TSIZE_C);
  /* gaspi_verify_queue_depth(glb_gaspi_ctx.ne_count_c[queue]); */

  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context const * const gctx = &glb_gaspi_ctx;

  if(lock_gaspi_tout (&gctx->lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if( eret != GASPI_SUCCESS)
	{
	  goto endL;
	}
    }

  eret = pgaspi_dev_gpu_write(segment_id_local, offset_local, rank,
			      segment_id_remote,offset_remote, (unsigned int) size,
			      queue, timeout_ms);

  if( eret != GASPI_SUCCESS )
    {
      /* gctx->qp_state_vec[queue][rank] = GASPI_STATE_CORRUPT; */
      goto endL;
    }

  /* GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_WRITE, 1); */
  /* GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_BYTES_WRITE, size); */

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;
}

#pragma weak gaspi_gpu_write_notify = pgaspi_gpu_write_notify
gaspi_return_t
pgaspi_gpu_write_notify(const gaspi_segment_id_t segment_id_local,
			const gaspi_offset_t offset_local,
			const gaspi_rank_t rank,
			const gaspi_segment_id_t segment_id_remote,
			const gaspi_offset_t offset_remote,
			const gaspi_size_t size,
			const gaspi_notification_id_t notification_id,
			const gaspi_notification_t notification_value,
			const gaspi_queue_id_t queue,
			const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_gpu_write_notify");
  gaspi_verify_local_off(offset_local, segment_id_local, size);
  gaspi_verify_remote_off(offset_remote, segment_id_remote, rank, size);
  gaspi_verify_queue(queue);
  gaspi_verify_comm_size(size, segment_id_local, segment_id_remote, rank, GASPI_MAX_TSIZE_C);
  /* gaspi_verify_queue_depth(glb_gaspi_ctx.ne_count_c[queue]); */

  if( notification_value == 0 )
    {
      gaspi_printf("Zero is not allowed as notification value.");
      return GASPI_ERR_INV_NOTIF_VAL;
    }

  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context const * const gctx = &glb_gaspi_ctx;

  if(lock_gaspi_tout (&gctx->lockC[queue], timeout_ms))
    return GASPI_TIMEOUT;

  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[rank].cstat )
    {
      eret = pgaspi_connect((gaspi_rank_t) rank, timeout_ms);
      if ( eret != GASPI_SUCCESS)
	{
	  goto endL;
	}
    }

  eret = pgaspi_dev_gpu_write_notify(segment_id_local, offset_local, rank,
				     segment_id_remote, offset_remote, size,
				     notification_id, notification_value,
				     queue, timeout_ms);
  if( eret != GASPI_SUCCESS )
    {
      /* gctx->qp_state_vec[queue][rank] = GASPI_STATE_CORRUPT; */
      goto endL;
    }

  /* GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_WRITE_NOT, 1); */
  /* GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_BYTES_WRITE, size); */

 endL:
  unlock_gaspi (&gctx->lockC[queue]);
  return eret;

}
