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

#ifndef GASPI_GPU_H
#define GASPI_GPU_H

#include <GASPI.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define  GASPI_GPU_MAX_SEG (224*1024*1024) 
  
  typedef int gaspi_gpu_t; 
  typedef int gaspi_gpu_num;

  gaspi_return_t gaspi_init_GPUs();
  gaspi_return_t gaspi_GPU_ids(gaspi_gpu_t *);
  gaspi_return_t gaspi_number_of_GPUs(gaspi_gpu_num *);
  
  gaspi_return_t gaspi_gpu_write(const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_rank_t,
				 const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_size_t,
				 const gaspi_queue_id_t, const gaspi_timeout_t);
  
  gaspi_return_t gaspi_gpu_write_notify(const gaspi_segment_id_t, const gaspi_offset_t, const gaspi_rank_t ,
					const gaspi_segment_id_t, const gaspi_offset_t,const gaspi_size_t,
					const gaspi_notification_id_t, const gaspi_notification_t,
					const gaspi_queue_id_t, const gaspi_timeout_t);
  
#ifdef __cplusplus
}
#endif

#endif // GASPI_GPU_H
