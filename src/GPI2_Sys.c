/*
Copyright (c) Fraunhofer ITWM, 2013-2026

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

#include <dirent.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#if defined(__x86_64__)
#include <cpuid.h>
#endif
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

/* Read the nominal TSC frequency (in MHz) directly from the hardware. Returns
 * 0.0f when it cannot be determined (caller falls back). */
static float
_gaspi_tsc_freq_cpuid (void)
{
#if defined(__x86_64__)
  unsigned int eax, ebx, ecx, edx;

  if (!__get_cpuid (0x80000007, &eax, &ebx, &ecx, &edx) || !(edx & (1u << 8)))
  {
    return 0.0f;
  }

  const unsigned int maxleaf = __get_cpuid_max (0, NULL);

  /* Leaf 0x15: TSC = core_crystal(ECX) * ratio_num(EBX) / ratio_den(EAX). */
  if (maxleaf >= 0x15)
  {
    __cpuid (0x15, eax, ebx, ecx, edx);
    if (eax != 0 && ebx != 0 && ecx != 0)
    {
      return (float) ((double) ecx * (double) ebx / (double) eax / 1.0e6);
    }
  }

  /* Leaf 0x16: processor base frequency (MHz), ~= TSC nominal on these
     parts. */
  if (maxleaf >= 0x16)
  {
    __cpuid (0x16, eax, ebx, ecx, edx);
    const unsigned int base_mhz = eax & 0xffff;
    if (base_mhz != 0)
    {
      return (float) base_mhz;
    }
  }
#endif
  return 0.0f;
}

/* Calibrate the cycle-counter frequency (MHz) from a few short busy-wait
 * windows timed with CLOCK_MONOTONIC_RAW. (no CPUID dependency) */

#define GASPI_CALIB_WINDOWS    5
#define GASPI_CALIB_WINDOW_NS  (5L * 1000L * 1000L)   /* 5 ms per window */

static float
_gaspi_calib_cpu_freq (void)
{
  double mhz[GASPI_CALIB_WINDOWS];

  for (int r = 0; r < GASPI_CALIB_WINDOWS; ++r)
  {
    struct timespec ts;

    if (clock_gettime (CLOCK_MONOTONIC_RAW, &ts))
    {
      return 0.0f;
    }

    const double t0 = (double) ts.tv_sec * 1.0e9 + (double) ts.tv_nsec;
    const gaspi_cycles_t c0 = gaspi_get_cycles();
    const double target = t0 + (double) GASPI_CALIB_WINDOW_NS;

    double t1;
    do
    {
      if (clock_gettime (CLOCK_MONOTONIC_RAW, &ts))
      {
        return 0.0f;
      }
      t1 = (double) ts.tv_sec * 1.0e9 + (double) ts.tv_nsec;
    }
    while (t1 < target);

    const gaspi_cycles_t c1 = gaspi_get_cycles();

    mhz[r] = (double) (c1 - c0) / (t1 - t0) * 1.0e3;   /* cyc/ns * 1e3 = MHz */
  }

  for (int i = 1; i < GASPI_CALIB_WINDOWS; ++i)
  {
    const double key = mhz[i];
    int j = i - 1;

    while (j >= 0 && mhz[j] > key)
    {
      mhz[j + 1] = mhz[j];
      --j;
    }
    mhz[j + 1] = key;
  }

  return (float) mhz[GASPI_CALIB_WINDOWS / 2];
}

float
gaspi_get_cpufreq (void)
{
  float mhz = 0.0f;

  /* Fast path: derive the TSC frequency from CPUID (instant, no busy-wait). */
  mhz = _gaspi_tsc_freq_cpuid();
  if (mhz > 0.0f)
  {
    return mhz;
  }

  /* CPUID didn't tell us: calibrate cheaply. */
  mhz = _gaspi_calib_cpu_freq();
  if (mhz > 0.0f)
  {
    return mhz;
  }

  /* Last resort: regression. */
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

/* NUMA node of a NIC, from sysfs by its device name (e.g. cxi0, mlx5_0),
   or -1. */
int
gaspi_nic_numa_node (const char *name)
{
  static const char *const classes[] = { "cxi", "infiniband" };

  for (size_t i = 0; i < sizeof (classes) / sizeof (classes[0]); i++)
  {
    char path[256];
    snprintf (path, sizeof (path), "/sys/class/%s/%s/device/numa_node",
              classes[i], name);

    FILE *f = fopen (path, "r");
    if (f == NULL)
    {
      continue;
    }

    int node = -1;
    if (fscanf (f, "%d", &node) != 1)
    {
      node = -1;
    }
    fclose (f);

    return node;
  }

  return -1;
}

/* The NUMA nodes with a NIC (cxi or verbs device), in increasing order;
   returns their number. */
static int
gaspi_nic_numa_nodes (int nodes[GASPI_MAX_NUMAS])
{
  static const char *const classes[] = { "cxi", "infiniband" };
  int has_nic[GASPI_MAX_NUMAS] = { 0 };

  for (size_t i = 0; i < sizeof (classes) / sizeof (classes[0]); i++)
  {
    char path[64];
    snprintf (path, sizeof (path), "/sys/class/%s", classes[i]);

    DIR *d = opendir (path);
    if (d == NULL)
    {
      continue;
    }

    for (struct dirent * e; (e = readdir (d)) != NULL;)
    {
      if (e->d_name[0] == '.')
      {
        continue;
      }

      const int node = gaspi_nic_numa_node (e->d_name);
      if (node >= 0 && node < GASPI_MAX_NUMAS)
      {
        has_nic[node] = 1;
      }
    }
    closedir (d);
  }

  int n = 0;
  for (int node = 0; node < GASPI_MAX_NUMAS; node++)
  {
    if (has_nic[node])
    {
      nodes[n++] = node;
    }
  }

  return n;
}

/* The NUMA nodes with CPUs (nodes with memory only do not count), in
   increasing order; returns their number. */
static int
gaspi_cpu_numa_nodes (int nodes[GASPI_MAX_NUMAS])
{
  int n = 0;

  for (int node = 0; node < GASPI_MAX_NUMAS; node++)
  {
    cpu_set_t cpus;

    if (gaspi_get_affinity_mask (node, &cpus) == 0 && CPU_COUNT (&cpus) > 0)
    {
      nodes[n++] = node;
    }
  }

  return n;
}

/* The k-th NUMA node with CPUs, counting round-robin from the first node
   with a NIC. Returns -1 if there is none. */
int
gaspi_get_numa_node (const int k)
{
  int nodes[GASPI_MAX_NUMAS];
  const int n = gaspi_cpu_numa_nodes (nodes);

  if (n == 0)
  {
    return -1;
  }

  int nics[GASPI_MAX_NUMAS];
  int first = 0;

  if (gaspi_nic_numa_nodes (nics) > 0)
  {
    for (int i = 0; i < n; i++)
    {
      if (nodes[i] == nics[0])
      {
        first = i;
      }
    }
  }

  return nodes[(first + k) % n];
}

/* NUMA node holding all CPUs this thread may run on, or -1. */
int
gaspi_cpus_numa_node (void)
{
  cpu_set_t allowed;
  if (sched_getaffinity (0, sizeof (allowed), &allowed) != 0)
  {
    return -1;
  }

  int nodes[GASPI_MAX_NUMAS];
  const int n = gaspi_cpu_numa_nodes (nodes);

  for (int i = 0; i < n; i++)
  {
    cpu_set_t node_cpus, both;

    if (gaspi_get_affinity_mask (nodes[i], &node_cpus) == 0)
    {
      CPU_AND (&both, &allowed, &node_cpus);
      if (CPU_EQUAL (&both, &allowed))
      {
        return nodes[i];
      }
    }
  }

  return -1;
}

/* Whether this thread may run on fewer CPUs than are online: someone (a
   launcher, numactl, taskset, a cpuset) has bound it already. */
int
gaspi_affinity_is_bound (void)
{
  cpu_set_t allowed;

  if (sched_getaffinity (0, sizeof (allowed), &allowed) != 0)
  {
    return 0;
  }

  return CPU_COUNT (&allowed) < sysconf (_SC_NPROCESSORS_ONLN);
}

/* Whether cpu is the first hardware thread of its core. */
static int
gaspi_cpu_is_first_thread (const int cpu)
{
  char path[96];
  snprintf (path, sizeof (path),
            "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);

  FILE *f = fopen (path, "r");
  if (f == NULL)
  {
    return 1;
  }

  int first = cpu;
  if (fscanf (f, "%d", &first) != 1)
  {
    first = cpu;
  }
  fclose (f);

  return first == cpu;
}

/* Appends the CPUs of the given NUMA nodes to order, one CPU of each node
   in turn; only first hardware threads of cores, or only the others. */
static void
gaspi_append_cpus (const int *nodes, const int n, const int first_threads,
                   int *order, int *count)
{
  cpu_set_t cpus[GASPI_MAX_NUMAS];
  int next[GASPI_MAX_NUMAS];

  for (int i = 0; i < n; i++)
  {
    if (gaspi_get_affinity_mask (nodes[i], &cpus[i]) != 0)
    {
      CPU_ZERO (&cpus[i]);
    }
    next[i] = 0;
  }

  for (int added = 1; added;)
  {
    added = 0;

    for (int i = 0; i < n; i++)
    {
      while (next[i] < CPU_SETSIZE)
      {
        const int cpu = next[i]++;

        if (CPU_ISSET (cpu, &cpus[i])
            && gaspi_cpu_is_first_thread (cpu) == first_threads)
        {
          order[(*count)++] = cpu;
          added = 1;
          break;
        }
      }
    }
  }
}

/* Binds this thread to one CPU, for communication benchmarks: local rank k
   gets the k-th CPU of the NUMA nodes with a NIC (one core of each such
   node in turn, cores before their second hardware threads), then of the
   other nodes. Returns the CPU or -1. */
int
gaspi_bind_nic_core (const int local_rank)
{
  int nics[GASPI_MAX_NUMAS], nodes[GASPI_MAX_NUMAS], others[GASPI_MAX_NUMAS];
  const int nnics = gaspi_nic_numa_nodes (nics);
  const int nnodes = gaspi_cpu_numa_nodes (nodes);
  int nothers = 0;

  for (int i = 0; i < nnodes; i++)
  {
    int is_nic = 0;

    for (int j = 0; j < nnics; j++)
    {
      is_nic |= nodes[i] == nics[j];
    }

    if (!is_nic)
    {
      others[nothers++] = nodes[i];
    }
  }

  static int order[CPU_SETSIZE];
  int count = 0;

  gaspi_append_cpus (nics, nnics, 1, order, &count);
  gaspi_append_cpus (nics, nnics, 0, order, &count);
  gaspi_append_cpus (others, nothers, 1, order, &count);
  gaspi_append_cpus (others, nothers, 0, order, &count);

  if (count == 0)
  {
    return -1;
  }

  const int cpu = order[local_rank % count];
  cpu_set_t one;

  CPU_ZERO (&one);
  CPU_SET (cpu, &one);
  if (sched_setaffinity (0, sizeof (one), &one) != 0)
  {
    return -1;
  }

  return cpu;
}

char *
pgaspi_gethostname (gaspi_context_t const *const gctx, const unsigned int id)
{
  return gctx->topology->hosts[id];
}

int
pgaspi_ranks_are_local (gaspi_rank_t a, gaspi_rank_t b)
{
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;
  return gctx->topology->hosts_ids[a] == gctx->topology->hosts_ids[b];
}

void gaspi_delay (void)
{
  GASPI_DELAY();
}
