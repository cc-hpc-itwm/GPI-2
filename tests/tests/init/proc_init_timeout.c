#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{

  TSUITE_INIT(argc, argv);
    
  ASSERT (gaspi_proc_init(5000));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}

