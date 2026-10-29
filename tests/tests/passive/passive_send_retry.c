#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

//a passive send retried after a timeout is resumed, not posted again: the
//late receiver gets the large message exactly once

#define MSG_SIZE 60000
#define SEND_TIMEOUT_MS 100
#define DUP_TIMEOUT_MS 2000

int
main (int argc, char *argv[])
{
  gaspi_rank_t nprocs, myrank;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));
  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  if (nprocs < 2)
  {
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  ASSERT (gaspi_segment_create (0, _1MB, GASPI_GROUP_ALL, GASPI_BLOCK,
                                GASPI_MEM_INITIALIZED));

  gaspi_pointer_t ptr;
  ASSERT (gaspi_segment_ptr (0, &ptr));
  unsigned char *const buf = (unsigned char *) ptr;

  //connect first: posting the first send may wait for the receiver
  if (myrank == 0)
  {
    ASSERT (gaspi_passive_send (0, 0, 1, sizeof (int), GASPI_BLOCK));
  }
  else if (myrank == 1)
  {
    gaspi_rank_t sender;

    ASSERT (gaspi_passive_receive (0, 0, &sender, sizeof (int),
                                   GASPI_BLOCK));
    assert (sender == 0);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (myrank == 0)
  {
    for (int i = 0; i < MSG_SIZE; i++)
    {
      buf[i] = (unsigned char) (i % 251);
    }

    int timeouts = 0;
    gaspi_return_t ret;

    while ((ret = gaspi_passive_send (0, 0, 1, MSG_SIZE, SEND_TIMEOUT_MS))
           == GASPI_TIMEOUT)
    {
      timeouts++;
    }

    fprintf (stderr, "send: %d after %d timeouts\n", ret, timeouts);
    assert (ret == GASPI_SUCCESS);
  }
  else if (myrank == 1)
  {
    gaspi_rank_t sender;

    sleep (1);

    ASSERT (gaspi_passive_receive (0, 0, &sender, MSG_SIZE, GASPI_BLOCK));
    assert (sender == 0);

    for (int i = 0; i < MSG_SIZE; i++)
    {
      assert (buf[i] == (unsigned char) (i % 251));
    }

    //a second copy of the message
    const gaspi_return_t ret =
      gaspi_passive_receive (0, 0, &sender, MSG_SIZE, DUP_TIMEOUT_MS);

    fprintf (stderr, "second receive: %d\n", ret);
    assert (ret == GASPI_TIMEOUT);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
