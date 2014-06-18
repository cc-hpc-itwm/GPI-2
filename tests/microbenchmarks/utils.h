#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

typedef unsigned long mcycles_t;

static inline mcycles_t
get_mcycles ()
{
  unsigned low, high;
  unsigned long long val;

  asm volatile ("rdtsc":"=a" (low), "=d" (high));
  val = high;
  val = (val << 32) | low;
  return val;
}

static int
mcycles_compare (const void *aptr, const void *bptr)
{
  const mcycles_t *a = (mcycles_t *) aptr;
  const mcycles_t *b = (mcycles_t *) bptr;
  if (*a < *b)
    return -1;
  if (*a > *b)
    return 1;
  return 0;
}

mcycles_t stamp[1024], stamp2[1024], delta[1024];

#endif //_UTILS_H_
