#include <stdio.h>
#include <stdlib.h>
#include <GASPI_Threads.h>

#include "test_utils.h"

void *
hello_fun (void *dummy)
{
  gaspi_int tid, total;

  ASSERT (gaspi_threads_register (&tid));

  ASSERT (gaspi_threads_get_total (&total));
  printf ("Hello from thread %d of %d\n", tid, total);

  gaspi_threads_sync ();

  return NULL;
}

int main()
{
  int i;
  int n;

  ASSERT (gaspi_threads_init (&n));

  for (i = 1; i < n; i++)
    ASSERT (gaspi_threads_run (hello_fun, NULL));

  hello_fun (NULL);

  ASSERT (gaspi_threads_term ());

  return EXIT_SUCCESS;
}
