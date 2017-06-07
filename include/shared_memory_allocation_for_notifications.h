#include <GASPI.h>

gaspi_return_t get_ptr_to_shared_notification_area
  (void** ptr_ptr_to_shared_notification_area, int* shm_fd);

gaspi_return_t free_shared_notification_area
  (void* ptr_to_shared_notification_area, int shm_fd);
