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

#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include "GASPI_types.h"
#include "GPI2.h"
#include "GPI2_Sys.h"
#include "GPI2_Types.h"

#define MEASUREMENTS 500
#define USECSTEP 10
#define USECSTART 100

/* CPU frequency through sampling and linear regression */
float
_gaspi_sample_cpu_freq (void)
{
  struct timeval tval1, tval2;
  gaspi_cycles_t start;
  double sumx = 0.0f;
  double sumy = 0.0f;
  double sum_sqr_x = 0.0f;
  double sum_sqr_y = 0.0f;
  double sumxy = 0.0f;
  double tx, ty;

  long x[MEASUREMENTS];
  gaspi_cycles_t y[MEASUREMENTS];
  double beta;
  double corr_2;

  for (int i = 0; i < MEASUREMENTS; ++i)
  {
    start = gaspi_get_cycles();

    if (gettimeofday (&tval1, NULL))
    {
      return 0.0f;
    }

    do
    {
      if (gettimeofday (&tval2, NULL))
      {
        return 0.0f;
      }
    }
    while ((tval2.tv_sec - tval1.tv_sec) * 1000000 +
           (tval2.tv_usec - tval1.tv_usec) < USECSTART + i * USECSTEP);


    x[i] =
      (tval2.tv_sec - tval1.tv_sec) * 1000000 + tval2.tv_usec - tval1.tv_usec;
    y[i] = gaspi_get_cycles() - start;
  }

  for (int i = 0; i < MEASUREMENTS; ++i)
  {
    tx = x[i];
    ty = y[i];
    sumx += tx;
    sumy += ty;
    sum_sqr_x += tx * tx;
    sum_sqr_y += ty * ty;
    sumxy += tx * ty;
  }

  corr_2 =
    (MEASUREMENTS * sumxy - sumx * sumy) *
    (MEASUREMENTS * sumxy - sumx * sumy)
     / (MEASUREMENTS * sum_sqr_x - sumx * sumx)
     / (MEASUREMENTS * sum_sqr_y - sumy * sumy);

  if (corr_2 < 0.9)
  {
    return 0.0f;
  }

  beta =
    (MEASUREMENTS * sumxy - sumx * sumy) / (MEASUREMENTS * sum_sqr_x -
                                            sumx * sumx);

  return (float) beta;
}

float
gaspi_get_cpufreq (void)
{
  float mhz = 0.0f;

  mhz = _gaspi_sample_cpu_freq();

  if (0.0f == mhz)
  {
    FILE *f;
    char buf[256];

    f = fopen ("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (f)
    {
      if (fgets (buf, sizeof (buf), f))
      {
        uint m;
        int rc;

        rc = sscanf (buf, "%u", &m);
        if (rc == 1)
        {
          mhz = (float) m;
        }
      }

      fclose (f);
    }

    if (mhz > 0.0f)
    {
      return mhz / 1000.0f;
    }

    f = fopen ("/proc/cpuinfo", "r");
    if (f)
    {
      while (fgets (buf, sizeof (buf), f))
      {
        float m;
        int rc;

        rc = sscanf (buf, "cpu MHz : %f", &m);

        if (rc != 1)
        {
          continue;
        }

        if (mhz == 0.0f)
        {
          mhz = m;
          continue;
        }
      }

      fclose (f);
    }
  }

  return mhz;
}

int
gaspi_get_affinity_mask (const int sock, cpu_set_t * cpuset)
{
  int rc;
  char buf[1024];
  unsigned int m[256];
  char path[256];

  memset (buf, 0, 1024);
  memset (m, 0, 256 * sizeof (unsigned int));

  snprintf (path, 256, "/sys/devices/system/node/node%d/cpumap", sock);

  FILE *f = fopen (path, "r");

  if (f == NULL)
  {
    return -1;
  }

  //read cpumap
  int id = 0;

  if (fgets (buf, sizeof (buf), f))
  {
    char *bptr = buf;

    while (1)
    {
      int ret = sscanf (bptr, "%x", &m[id]);

      if (ret <= 0)
      {
        break;
      }

      int found = 0;
      unsigned int cpos = 0;

      size_t length = strlen (bptr);

      for (size_t j = 0; j < length - 1; j++)
      {
        if (bptr[j] == ',')
        {
          found = 1;
          break;
        }
        cpos++;
      }

      if (!found)
      {
        if ((cpos + 1) > strlen (bptr))
        {
          fclose (f);
          return -1;
        }
      }

      bptr += (cpos + 1);
      id++;
    }
  }

  rc = id;

  memset (cpuset, 0, sizeof (cpu_set_t));

  char *ptr = (char *) cpuset;
  int pos = 0;

  for (int i = rc - 1; i >= 0; i--)
  {
    memcpy (ptr + pos, &m[i], sizeof (unsigned int));
    pos += sizeof (unsigned int);
  }

  fclose (f);

  return 0;
}
