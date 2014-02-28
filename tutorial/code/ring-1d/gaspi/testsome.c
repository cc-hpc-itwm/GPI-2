#include "testsome.h"

#include "assert.h"
#include "success_or_die.h"

int test_or_die
  ( gaspi_segment_id_t segment_id
  , gaspi_notification_id_t notification_id
  , gaspi_notification_t expected
  )
{
  gaspi_notification_id_t id;
  gaspi_return_t ret;

  if ( ( ret =
         gaspi_notify_waitsome (segment_id, notification_id, 1, &id, GASPI_TEST)
       ) == GASPI_SUCCESS
     )
  {
    ASSERT (id == notification_id);

    gaspi_notification_t value;

    SUCCESS_OR_DIE (gaspi_notify_reset (segment_id, id, &value));

    ASSERT (value == expected);

    return 1;
  }
  else
  {
    ASSERT (ret != GASPI_ERROR);

    return 0;
  }
}
