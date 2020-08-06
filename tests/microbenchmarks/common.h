#ifndef _BENCH_COMMON_H_
#define _BENCH_COMMON_H_

#include <GASPI.h>

#define ITERATIONS 1000

#define MAX(a,b) (((a)<(b)) ? (b) : (a))
#define MIN(a,b) (((a)>(b)) ? (b) : (a))

#define GPI2_ASSERT(s)                                                  \
  if(s != GASPI_SUCCESS)                                                \
  {                                                                     \
    gaspi_printf ("GASPI error:" #s " %d\n",__LINE__);                  \
    _exit (EXIT_FAILURE);                                               \
  }

gaspi_cycles_t stamp[ITERATIONS], stamp2[ITERATIONS], delta[ITERATIONS];

int mcycles_compare (const void*, const void*);

int start_bench (int);

void end_bench (void);

#endif //_BENCH_COMMON_H_
