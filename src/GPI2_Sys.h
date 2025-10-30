/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2024

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

#ifndef GPI2_SYS_H
#define GPI2_SYS_H 1

#include <sched.h>

#include "GASPI_types.h"

#if defined(__x86_64__)
#include <xmmintrin.h>
#define GASPI_DELAY() _mm_pause()

#elif defined(__aarch64__)
#define GASPI_DELAY()  __asm__ __volatile__("yield")

#elif defined (__PPC64__)
#define GASPI_DELAY() __asm__ volatile("ori 0,0,0" ::: "memory");
#endif

static inline gaspi_cycles_t
gaspi_get_cycles (void)
{
#if defined(__x86_64__)

  unsigned low, high;
  unsigned long long val;

  __asm__ volatile ("rdtsc":"=a" (low), "=d" (high));

  val = high;
  val = (val << 32) | low;
  return val;

#elif defined(__aarch64__)

  unsigned long ts;
  asm volatile ("isb; mrs %0, cntvct_el0" : "=r" (ts));
  return ts;

#elif defined (__PPC64__)

  unsigned long cycles;
  asm volatile ("mftb %0" : "=r" (cycles) : );
  return cycles;

#endif
}

float gaspi_get_cpufreq (void);

int gaspi_get_affinity_mask (const int sock, cpu_set_t * cpuset);

char *pgaspi_gethostname (const unsigned int id);

int pgaspi_ranks_are_local (gaspi_rank_t a, gaspi_rank_t b);

#endif //GPI2_SYS
