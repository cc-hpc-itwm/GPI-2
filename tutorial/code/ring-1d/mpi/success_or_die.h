#ifndef SUCCESS_OR_DIE_H
#define SUCCESS_OR_DIE_H

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

#define SUCCESS_OR_DIE(f...)                                            \
  do                                                                    \
  {                                                                     \
    const int r = f;                                                    \
                                                                        \
    if (r != MPI_SUCCESS)                                               \
    {                                                                   \
      printf ("Error: '%s' [%s:%i]: %i\n", #f, __FILE__, __LINE__, r);  \
                                                                        \
      exit (EXIT_FAILURE);                                              \
    }                                                                   \
  } while (0)

#endif
