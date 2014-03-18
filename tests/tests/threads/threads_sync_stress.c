#include <stdio.h>
#include <stdlib.h>
#include <GPI2_Threads.h>

#include "test_utils.h"

#define ITERATIONS 1000000
void * hello_fun(void * dummy)
{
  int i;
  gaspi_int tid, total;
  ASSERT (gaspi_threads_register(&tid));

  ASSERT (gaspi_threads_get_total(&total));
  printf("Hello from thread %d of %d\n", 
	 tid,
	 total);

  for(i = 0; i < ITERATIONS; i++)
	gaspi_threads_sync();

  return NULL;
}

int main(int argc, char * argv[])
{
  int i;
  int n;
  ASSERT(gaspi_threads_init(&n));

  for(i = 1; i < n; i++)
    ASSERT (gaspi_threads_run(hello_fun, NULL));

  hello_fun(NULL);

  ASSERT( gaspi_threads_term());

  return EXIT_SUCCESS;
}
