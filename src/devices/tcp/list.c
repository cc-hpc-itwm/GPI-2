/*
Copyright (c) Fraunhofer ITWM, 2013-2025

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

#include <stdlib.h>
#include "list.h"
#include "tcp_device.h"

inline void
list_insert (list * l, const tcp_dev_wr_t * wr)
{
  listNode *node = calloc (1, sizeof (listNode));

  node->wr = *wr;

  if (l->last == NULL)
  {
    l->first = l->last = node;
  }
  else
  {
    l->last->next = node;
    node->prev = l->last;
    l->last = node;
  }

  l->count++;
}

inline void
list_remove (list * l, listNode * node)
{
  if (l->count == 0 || node == NULL)
  {
    return;
  }

  if (node == l->first && node == l->last)
  {
    l->first = l->last = NULL;
  }
  else if (node == l->first)
  {
    l->first = node->next;
    l->first->prev = NULL;
  }
  else if (node == l->last)
  {
    l->last = node->prev;
    l->last->next = NULL;
  }
  else
  {
    listNode *after = node->next;
    listNode *before = node->prev;

    after->prev = before;
    before->next = after;
  }

  l->count--;

  free (node);
  node = NULL;
}

inline void
list_clear (list * l)
{
  while (l->count > 0)
  {
    list_remove (l, l->first);
  }
}
