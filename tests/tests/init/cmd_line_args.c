#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{

  assert ((argc > 1));
  
  ASSERT (gaspi_proc_init(5000));

  ASSERT (gaspi_proc_term(5000));

  return EXIT_SUCCESS;
}
