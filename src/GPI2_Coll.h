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

#ifndef _GPI2_COLL_H_
#define _GPI2_COLL_H_

#include <stdlib.h>

#include "GASPI.h"

typedef enum
  {
    GASPI_OP,
    GASPI_USER
  }redux_type_t;

struct redux_args
{
  redux_type_t f_type;
  union
  {
    struct
    {
      gaspi_size_t elem_size;
      gaspi_operation_t op;
      gaspi_datatype_t type;
    };

    struct
    {
      gaspi_size_t elem_size;
      gaspi_reduce_operation_t user_fct;
      gaspi_state_t rstate;
    } ;
  } f_args;
};

void (*fctArrayGASPI[18]) (void *, void *, void *, const unsigned char cnt);

void
gaspi_init_collectives ();

#endif //_GPI2_COLL_H_
