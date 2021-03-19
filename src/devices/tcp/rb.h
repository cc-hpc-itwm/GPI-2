/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2021

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
#ifndef _GPI2_RB_H_
#define _GPI2_RB_H_

typedef struct
{
  volatile unsigned long seq;
  void *data;
} rb_cell;

struct ringbuffer
{
  rb_cell *cells;

  volatile unsigned long mask;
  volatile unsigned long ipos;
  volatile unsigned long rpos;
} __attribute__ ((aligned (64)));

typedef struct ringbuffer ringbuffer;

int insert_ringbuffer (ringbuffer * rb, void *data);
int remove_ringbuffer (ringbuffer * rb, void **data);

#endif
