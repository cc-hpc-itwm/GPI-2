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

#ifndef GPI2_GPU_H
#define GPI2_GPU_H

#include "GPI2.h"
#include <cuda_runtime.h>
#define GASPI_CUDA_EVENTS 2
#define GASPI_GPU_DIRECT_MAX (32*1024)
#define GASPI_GPU_BUFFERED  (128*1024)

typedef struct {
  unsigned long offset_local, offset_remote, size;
  gaspi_rank_t rank;
  gaspi_segment_id_t segment_local, segment_remote;
  cudaEvent_t event;
  int ib_use;
  int in_use;

gaspi_cuda_event events[GASPI_MAX_QP][GASPI_CUDA_EVENTS];
}
gaspi_cuda_event;


typedef struct{
  char gpu_direct;
  int device_id;
  cudaStream_t streams[GASPI_MAX_QP];
  char name[256];
}gaspi_gpu;

gaspi_gpu *gpus;



//TODO: need to move this somewhere else
static gaspi_gpu* _gaspi_find_gpu(int dev_id)
{
  int i;
  for (i = 0; i < glb_gaspi_ctx.gpu_count; i++)
    if(gpus[i].device_id == dev_id)
      {
	cudaSetDevice(dev_id);
	return &gpus[i];
      }
  
  return NULL;
}

#endif //GPI2_GPU_H_
