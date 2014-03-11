/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013

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
#include "GPI2_Utility.h"

ulong
gaspi_load_ulong(volatile ulong *ptr)
{
  ulong v=*ptr;
  asm volatile("" ::: "memory");
  return v;
}

//TODO: sample it instead of trusting OS? like mctp?
float
gaspi_get_cpufreq ()
{
  FILE *f;
  char buf[256];
  float mhz = 0.0f;

  f = fopen ("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
  if (f)
    {
      if (fgets (buf, sizeof (buf), f))
	{
	  uint m;
	  int rc;

	  rc = sscanf (buf, "%u", &m);
	  if (rc == 1)
	    mhz = (float) m;
	}

      fclose (f);
    }

  if (mhz > 0.0f)
    return mhz / 1000.0f;

  f = fopen ("/proc/cpuinfo", "r");

  if (f)
    {
      while (fgets (buf, sizeof (buf), f))
	{
	  float m;
	  int rc;

	  rc = sscanf (buf, "cpu MHz : %f", &m);

	  if (rc != 1)
	    continue;

	  if (mhz == 0.0f)
	    {
	      mhz = m;
	      continue;
	    }
	}

      fclose (f);
    }

  return mhz;
}

int
gaspi_get_affinity_mask (const int sock, cpu_set_t * cpuset)
{

  int i, rc;
  char buf[1024];
  unsigned int m[256];
  char path[256];
  FILE *f;

  memset (buf, 0, 1024);
  memset (m, 0, 256 * sizeof (unsigned int));

  snprintf (path, 256, "/sys/devices/system/node/node%d/cpumap", sock);

  f = fopen (path, "r");

  if (!f)
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
	    break;
	  bptr += 9;
	  id++;
	}
    }

  rc = id;

  memset (cpuset, 0, sizeof (cpu_set_t));
  char *ptr = (char *) cpuset;

  int pos = 0;

  for (i = rc - 1; i >= 0; i--)
    {
      memcpy (ptr + pos, &m[i], sizeof (unsigned int));
      pos += sizeof (unsigned int);
    }

  fclose (f);
  return 0;
}


void
gaspi_thread_sleep(int msecs)
{
  struct timespec sleep_time, rem;
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = msecs * 10000;
  nanosleep(&sleep_time, &rem);
}
