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

#ifndef _GPI2_LIST_H_
#define _GPI2_LIST_H_

#include "tcp_device.h"

typedef struct listNode 
{
  struct listNode *prev, *next;
  tcp_dev_wr_t wr;
} listNode;


typedef struct list 
{
  struct listNode *first, *last;
  int count;
} list;

void list_remove(list *l, listNode *node);
void list_insert(list *l, const tcp_dev_wr_t *wr);
void list_clear(list *l);

#endif /* _GPI2_LIST_H_ */
