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
#include "GPI2_Coll.h"
#include "GPI2_Utility.h"

/* Define the array of pre-defined collective operations */
void (*fctArrayGASPI[GASPI_COLL_OP_TYPES]) (void *, void *, void *,
                                            const int cnt);

//pre-defined coll. operations
void
opMinIntGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  int *rv = (int *) res;
  int *lv = (int *) localVal;
  int *dv = (int *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MIN (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opMaxIntGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  int *rv = (int *) res;
  int *lv = (int *) localVal;
  int *dv = (int *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MAX (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opSumIntGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  int *rv = (int *) res;
  int *lv = (int *) localVal;
  int *dv = (int *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = *lv + *dv;
    lv++;
    dv++;
    rv++;
  }
}

void
opMinUIntGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  unsigned int *rv = (unsigned int *) res;
  unsigned int *lv = (unsigned int *) localVal;
  unsigned int *dv = (unsigned int *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MIN (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opMaxUIntGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  unsigned int *rv = (unsigned int *) res;
  unsigned int *lv = (unsigned int *) localVal;
  unsigned int *dv = (unsigned int *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MAX (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opSumUIntGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  unsigned int *rv = (unsigned int *) res;
  unsigned int *lv = (unsigned int *) localVal;
  unsigned int *dv = (unsigned int *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = *lv + *dv;
    lv++;
    dv++;
    rv++;
  }
}

void
opMinFloatGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  float *rv = (float *) res;
  float *lv = (float *) localVal;
  float *dv = (float *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MIN (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opMaxFloatGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  float *rv = (float *) res;
  float *lv = (float *) localVal;
  float *dv = (float *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MAX (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opSumFloatGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  float *rv = (float *) res;
  float *lv = (float *) localVal;
  float *dv = (float *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = *lv + *dv;
    lv++;
    dv++;
    rv++;
  }
}

void
opMinDoubleGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  double *rv = (double *) res;
  double *lv = (double *) localVal;
  double *dv = (double *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MIN (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opMaxDoubleGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  double *rv = (double *) res;
  double *lv = (double *) localVal;
  double *dv = (double *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MAX (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opSumDoubleGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  double *rv = (double *) res;
  double *lv = (double *) localVal;
  double *dv = (double *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = *lv + *dv;
    lv++;
    dv++;
    rv++;
  }
}

void
opMinLongGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  long *rv = (long *) res;
  long *lv = (long *) localVal;
  long *dv = (long *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MIN (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opMaxLongGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  long *rv = (long *) res;
  long *lv = (long *) localVal;
  long *dv = (long *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MAX (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opSumLongGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  long *rv = (long *) res;
  long *lv = (long *) localVal;
  long *dv = (long *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = *lv + *dv;
    lv++;
    dv++;
    rv++;
  }
}

void
opMinULongGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  unsigned long *rv = (unsigned long *) res;
  unsigned long *lv = (unsigned long *) localVal;
  unsigned long *dv = (unsigned long *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MIN (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opMaxULongGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  unsigned long *rv = (unsigned long *) res;
  unsigned long *lv = (unsigned long *) localVal;
  unsigned long *dv = (unsigned long *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = MAX (*lv, *dv);
    lv++;
    dv++;
    rv++;
  }
}

void
opSumULongGASPI (void *res, void *localVal, void *dstVal, const int cnt)
{
  unsigned long *rv = (unsigned long *) res;
  unsigned long *lv = (unsigned long *) localVal;
  unsigned long *dv = (unsigned long *) dstVal;

  for (int i = 0; i < cnt; i++)
  {
    *rv = *lv + *dv;
    lv++;
    dv++;
    rv++;
  }
}

void
gaspi_init_collectives (void)
{
  fctArrayGASPI[0] = &opMinIntGASPI;
  fctArrayGASPI[1] = &opMinUIntGASPI;
  fctArrayGASPI[2] = &opMinFloatGASPI;
  fctArrayGASPI[3] = &opMinDoubleGASPI;
  fctArrayGASPI[4] = &opMinLongGASPI;
  fctArrayGASPI[5] = &opMinULongGASPI;
  fctArrayGASPI[6] = &opMaxIntGASPI;
  fctArrayGASPI[7] = &opMaxUIntGASPI;
  fctArrayGASPI[8] = &opMaxFloatGASPI;
  fctArrayGASPI[9] = &opMaxDoubleGASPI;
  fctArrayGASPI[10] = &opMaxLongGASPI;
  fctArrayGASPI[11] = &opMaxULongGASPI;
  fctArrayGASPI[12] = &opSumIntGASPI;
  fctArrayGASPI[13] = &opSumIntGASPI;
  fctArrayGASPI[14] = &opSumFloatGASPI;
  fctArrayGASPI[15] = &opSumDoubleGASPI;
  fctArrayGASPI[16] = &opSumLongGASPI;
  fctArrayGASPI[17] = &opSumULongGASPI;
}
