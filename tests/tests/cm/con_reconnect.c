#include <unistd.h>
#include <test_utils.h>

/* Without a static topology, three rounds: all ranks connect to all
   (ascending), then again after a disconnect (descending), then implicitly
   by the first write; each round every rank writes to every rank. */

#define ROUNDS 3

static void
round_traffic (int r, gaspi_rank_t myrank, gaspi_rank_t num)
{
  gaspi_pointer_t ptr;
  ASSERT (gaspi_segment_ptr (0, &ptr));
  unsigned long *const slot = (unsigned long *) ptr;

  slot[ROUNDS * num + myrank] = (unsigned long) (r * num + myrank + 1);

  for (gaspi_rank_t i = 0; i < num; i++)
  {
    const gaspi_size_t word = sizeof (unsigned long);

    ASSERT (gaspi_write_notify (0, (ROUNDS * num + myrank) * word,
                                i, 0, (r * num + myrank) * word, word,
                                (gaspi_notification_id_t) (r * num + myrank),
                                1, 0, GASPI_BLOCK));
    ASSERT (gaspi_wait (0, GASPI_BLOCK));
  }

  for (gaspi_rank_t n = 0; n < num; n++)
  {
    gaspi_notification_id_t id;
    gaspi_notification_t val;

    ASSERT (gaspi_notify_waitsome (0, (gaspi_notification_id_t) (r * num),
                                   num, &id, GASPI_BLOCK));
    ASSERT (gaspi_notify_reset (0, id, &val));
    assert (val == 1);
  }

  for (gaspi_rank_t i = 0; i < num; i++)
  {
    assert (slot[r * num + i] == (unsigned long) (r * num + i + 1));
  }
}

int
main (int argc, char *argv[])
{
  gaspi_config_t conf;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_config_get (&conf));
  conf.build_infrastructure = GASPI_TOPOLOGY_NONE;
  ASSERT (gaspi_config_set (conf));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t myrank, num;
  ASSERT (gaspi_proc_rank (&myrank));
  ASSERT (gaspi_proc_num (&num));

  gaspi_number_t notif_num;
  ASSERT (gaspi_notification_num (&notif_num));
  if (notif_num < (gaspi_number_t) ROUNDS * num)
  {
    TSUITE_SKIP ("too few notifications");
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  gaspi_group_t g;
  ASSERT (gaspi_group_create (&g));
  for (gaspi_rank_t i = 0; i < num; i++)
  {
    ASSERT (gaspi_group_add (g, i));
  }
  ASSERT (gaspi_group_commit (g, GASPI_BLOCK));
  ASSERT (gaspi_segment_create (0, (ROUNDS + 1) * num * sizeof (unsigned long),
                                g, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  for (gaspi_rank_t i = 0; i < num; i++)
  {
    ASSERT (gaspi_disconnect (i, GASPI_BLOCK));
  }
  sleep (1);

  for (int r = 0; r < ROUNDS; r++)
  {
    for (gaspi_rank_t n = 0; n < num && r < 2; n++)
    {
      const gaspi_rank_t i = r == 0 ? n : num - 1 - n;

      ASSERT (gaspi_connect (i, GASPI_BLOCK));
    }

    round_traffic (r, myrank, num);
    ASSERT (gaspi_barrier (g, GASPI_BLOCK));

    if (r < ROUNDS - 1)
    {
      for (gaspi_rank_t i = 0; i < num; i++)
      {
        ASSERT (gaspi_disconnect (i, GASPI_BLOCK));
      }
      sleep (1);
    }
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
