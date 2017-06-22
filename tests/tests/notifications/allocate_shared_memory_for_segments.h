#include <GASPI.h>

gaspi_return_t get_ptr_to_shared_data
  (void** ptr_ptr_to_shared_notification_area, unsigned long size, int* shm_fd, const char* const name);

gaspi_return_t free_shared_data
  (void* ptr_to_shared_notification_area, unsigned long size, int shm_fd, const char* const name);
