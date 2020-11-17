/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2020

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
} redux_type_t;

struct redux_args
{
  redux_type_t f_type;
  gaspi_size_t elem_size;
  gaspi_number_t elem_cnt;

  union
  {
    struct
    {
      gaspi_operation_t op;
      gaspi_datatype_t type;
    };

    struct
    {
      gaspi_reduce_operation_t user_fct;
      gaspi_reduce_state_t rstate;
    };
  } f_args;
};

/* Number of collectives possibilities (Types x Ops) */
#define GASPI_COLL_OP_TYPES 18

/* Collective functions' signatures */
void
opMinIntGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMaxIntGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opSumIntGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMinUIntGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMaxUIntGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opSumUIntGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMinFloatGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMaxFloatGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opSumFloatGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMinDoubleGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMaxDoubleGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opSumDoubleGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMinLongGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMaxLongGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opSumLongGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMinLongUGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opMaxLongUGASPI (void *res, void *localVal, void *dstVal, const int cnt);

void
opSumLongUGASPI (void *res, void *localVal, void *dstVal, const int cnt);

/* Declare the array of pre-defined collective operations */
extern void (*fctArrayGASPI[GASPI_COLL_OP_TYPES]) (void *, void *, void *,
                                            const int cnt);

void gaspi_init_collectives (void);

#endif //_GPI2_COLL_H_
