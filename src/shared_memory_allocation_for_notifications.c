#include <shared_memory_allocation_for_notifications.h>
#include <GASPI_Ext.h>
#include <../src/GPI2.h>

#include <assert.h>
#include <fcntl.h>
#include <semaphore.h>
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
      exit (-1);                                \
    }                                           \
  } while (0)

gaspi_return_t get_ptr_to_shared_notification_area
(void** ptr_ptr_to_shared_notification_area, int* shm_fd)
{
  gaspi_rank_t gaspi_local_rank;
  SUCCESS_OR_DIE (gaspi_proc_local_rank, &gaspi_local_rank);

  const char* const name = "/shared_notifications";

  const char* const sem_name = "/gpi2_notif";

  sem_t* sem = sem_open (sem_name, O_CREAT| O_RDWR, 0666, 0);

  if( sem == SEM_FAILED )
    {
      goto sem_error;
    }

  if( gaspi_local_rank != 0 && sem_wait (sem))
    {
      goto sem_error;
    }

  *shm_fd = shm_open (name, O_CREAT | O_RDWR, 0666);
  if( *shm_fd == -1 )
    {
      gaspi_print_error ("Shared memory for notifications failed!");
      return GASPI_ERROR;
    }

  if (ftruncate (*shm_fd, NOTIFY_OFFSET) )
    {
      gaspi_print_error ("Failed to truncate notifications space.");
      return GASPI_ERROR;
    }

  *ptr_ptr_to_shared_notification_area = mmap (0,
                                               NOTIFY_OFFSET,
                                               PROT_READ | PROT_WRITE,
                                               MAP_SHARED,
                                               *shm_fd,
                                               0);

  if( *ptr_ptr_to_shared_notification_area == MAP_FAILED )
    {
      gaspi_print_error ("Allocating shared memory for notifications failed");
      close (*shm_fd);
      shm_unlink (name);
      return GASPI_ERROR;
    }


  if( gaspi_local_rank == 0 )
    {
      gaspi_rank_t local_ranks;
      gaspi_proc_local_num (&local_ranks);
      while ( local_ranks > 0 )
        {
          if( sem_post (sem) )
          {
            goto sem_error;
          }

          local_ranks--;
        }
    }

  if( gaspi_local_rank != 0 && sem_close (sem))
    {
      goto sem_error;
    }

  if( gaspi_local_rank == 0 )
    {
      /* be the last one waiting on semaphore*/
      int v;
      do
        {
          sem_getvalue (sem, &v);
        }
      while (v != 1);

      if( sem_wait (sem)       ||
          sem_close (sem)      ||
          sem_unlink (sem_name) )
        {
          goto sem_error;
        }
    }

  return GASPI_SUCCESS;

 sem_error:
  gaspi_print_error ("Failed to synchronize ranks");
  return GASPI_ERROR;
}

gaspi_return_t free_shared_notification_area
  (void* ptr_to_shared_notification_area, int shm_fd)
{
  if(munmap (ptr_to_shared_notification_area, NOTIFY_OFFSET) == -1)
    {
      gaspi_print_error ("Unmapping shared memory for notifications failed!");
      return GASPI_ERROR;
    }

  if(close (shm_fd) == -1)
    {
      gaspi_print_error ("Closing the shared memory segment for notifications failed!");
      return GASPI_ERROR;
    }

  shm_unlink ("/shared_notifications");

  return GASPI_SUCCESS;
}
