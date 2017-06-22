#include "allocate_shared_memory_for_segments.h"

#include <GASPI_Ext.h>

#include <assert.h>
#include <fcntl.h>
#include <memory.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <unistd.h>

#define SUCCESS_OR_DIE(f, args...)              \
  do                                            \
  {                                             \
    gaspi_return_t const r = f (args);          \
                                                \
    if (r != GASPI_SUCCESS)                     \
    {                                           \
      gaspi_printf ( "Error[%s:%i]: %s\n"       \
                   , __FILE__                   \
                   , __LINE__                   \
                   , gaspi_error_str (r)        \
                   );                           \
                                                \
      exit (-1);                      \
    }                                           \
  } while (0)


gaspi_return_t get_ptr_to_shared_data
  (void** ptr_ptr_to_shared_notification_area, unsigned long size, int* shm_fd, const char* const name)
{
  gaspi_rank_t gaspi_local_rank;
  SUCCESS_OR_DIE (gaspi_proc_local_rank, &gaspi_local_rank);

  if (gaspi_local_rank == 0)
  {
    *shm_fd = shm_open (name, O_CREAT | O_RDWR, 0666);
    if (*shm_fd == -1)
     {
       gaspi_printf ("Shared memory for data failed!");
       return GASPI_ERROR;
     }

     ftruncate (*shm_fd, size);

     *ptr_ptr_to_shared_notification_area = mmap ( 0
                                                 , size, PROT_READ | PROT_WRITE
                                                 , MAP_SHARED
                                                 , *shm_fd
                                                 , 0
                                                 );

     if (*ptr_ptr_to_shared_notification_area == MAP_FAILED)
     {
       gaspi_printf ("Allocating shared memory for data failed");
       close (*shm_fd);
       shm_unlink (name);
       return GASPI_ERROR;
     }

     SUCCESS_OR_DIE (gaspi_barrier, GASPI_GROUP_ALL, GASPI_BLOCK) ;
  }
  else
  {
    SUCCESS_OR_DIE (gaspi_barrier, GASPI_GROUP_ALL, GASPI_BLOCK) ;

    *shm_fd = shm_open (name, O_RDWR, 0666);
    if (*shm_fd == -1)
    {
      gaspi_printf ("Opening shared memory for data failed!");
      return GASPI_ERROR;
    }

    *ptr_ptr_to_shared_notification_area = mmap ( 0
                                                , size
                                                , PROT_READ | PROT_WRITE
                                                , MAP_SHARED
                                                , *shm_fd
                                                , 0
                                                );

    if (*ptr_ptr_to_shared_notification_area == MAP_FAILED)
    {
      gaspi_printf  ("Mapping shared memory for data failed!");
      close (*shm_fd);
      shm_unlink (name);
      return GASPI_ERROR;
    }
  }

  return GASPI_SUCCESS;
}

gaspi_return_t free_shared_data
  (void* ptr_to_shared_notification_area, unsigned long size, int shm_fd, const char* const name)
{
  if (munmap (ptr_to_shared_notification_area, size) == -1)
  {
    gaspi_printf ("Unmapping shared memory for data failed!");
    return GASPI_ERROR;
  }

  if (close (shm_fd) == -1)
  {
    gaspi_printf ("Closing the shared memory segment for data failed!");
    return GASPI_ERROR;
  }

  shm_unlink (name);

  return GASPI_SUCCESS;
}
