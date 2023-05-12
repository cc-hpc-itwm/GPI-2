/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2023

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

#ifndef _GPI2_SEG_H_
#define _GPI2_SEG_H_ 1

typedef struct
{
  int rank;
  int ret;
  int seg_id;
  unsigned long addr;
  unsigned long size;
  unsigned long notif_addr;

#ifdef GPI2_DEVICE_IB
  int rkey[2];
#endif
} gaspi_segment_descriptor_t;


int gaspi_segment_set (const gaspi_segment_descriptor_t snp);

#endif //_GPI2_SEG_H_
