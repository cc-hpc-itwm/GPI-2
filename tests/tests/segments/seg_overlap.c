#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <test_utils.h>

/* Segments on overlapping parts of one user buffer, each case checked by a
   write_notify around the ring:
   1 - a segment inside another, written from the outer into the inner
   2 - the inner segment deleted, the outer one still used
   3 - the outer segment deleted, the inner one still used
   4 - two segments overlapping partly
   5 - an atomic on the inner segment of case 3 */

#define BUF_SIZE (4 * _1MB)
#define MSG_SIZE (64 * 1024)
#define SEG_TIMEOUT 5000

static gaspi_rank_t rank, nprocs, left, right;
static char *buf;
static int bad = 0;

static int
use (const gaspi_segment_id_t seg, const size_t off, const size_t size,
     const int c)
{
  const gaspi_return_t ret = gaspi_segment_use (seg, buf + off, size,
                                                GASPI_GROUP_ALL, SEG_TIMEOUT,
                                                0);
  if (ret != GASPI_SUCCESS)
  {
    fprintf (stderr, "case %d: segment_use %u at %zu (%zu bytes): %d\n",
             c, seg, off, size, ret);
    bad++;
  }

  return ret == GASPI_SUCCESS;
}

/* src -> dst of the right neighbour; checks the data from the left one. */
static void
transfer (const gaspi_segment_id_t src, const gaspi_offset_t src_off,
          const gaspi_segment_id_t dst, const gaspi_offset_t dst_off,
          const int c)
{
  gaspi_pointer_t p;
  gaspi_notification_id_t id;
  gaspi_notification_t val;

  ASSERT (gaspi_segment_ptr (src, &p));
  memset ((char *) p + src_off, rank * 16 + c, MSG_SIZE);

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_write_notify (src, src_off, right, dst, dst_off, MSG_SIZE,
                              (gaspi_notification_id_t) c, 1, 0,
                              GASPI_BLOCK));
  ASSERT (gaspi_notify_waitsome (dst, (gaspi_notification_id_t) c, 1, &id,
                                 GASPI_BLOCK));
  ASSERT (gaspi_notify_reset (dst, id, &val));
  ASSERT (gaspi_wait (0, GASPI_BLOCK));

  ASSERT (gaspi_segment_ptr (dst, &p));
  const unsigned char *const r = (unsigned char *) p + dst_off;
  int wrong = 0;
  for (int i = 0; i < MSG_SIZE; i++)
  {
    wrong += (r[i] != (unsigned char) (left * 16 + c));
  }

  if (wrong)
  {
    fprintf (stderr, "case %d: %d bytes wrong\n", c, wrong);
    bad++;
  }

  /* Nobody deletes a segment the neighbour still writes into. */
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
}

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));
  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  left = (rank + nprocs - 1) % nprocs;
  right = (rank + 1) % nprocs;

  assert (posix_memalign ((void **) &buf, 4096, BUF_SIZE) == 0);
  memset (buf, 0, BUF_SIZE);

  //1: inner [1 MB, 2 MB) inside outer [0, 4 MB)
  const int s0 = use (0, 0, BUF_SIZE, 1);
  const int s1 = s0 && use (1, _1MB, _1MB, 1);
  if (s1)
  {
    transfer (0, 0, 1, 0, 1);
  }

  //2: inner deleted, outer still used
  if (s1)
  {
    ASSERT (gaspi_segment_delete (1));
    transfer (0, 3 * _1MB, 0, 3 * _1MB - MSG_SIZE, 2);
  }

  //3: a new inner one, then the outer one deleted
  const int s2 = s0 && use (2, _1MB, _1MB, 3);
  if (s0)
  {
    ASSERT (gaspi_segment_delete (0));
  }
  if (s2)
  {
    transfer (2, 0, 2, MSG_SIZE, 3);
  }

  //5: an atomic on the inner segment (its outer one is deleted)
  if (s2)
  {
    const gaspi_offset_t off = 3 * MSG_SIZE;
    gaspi_pointer_t p;
    gaspi_atomic_value_t old;

    ASSERT (gaspi_segment_ptr (2, &p));
    *(gaspi_atomic_value_t *) ((char *) p + off) = 0;
    ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

    ASSERT (gaspi_atomic_fetch_add (2, off, right, 5, &old, GASPI_BLOCK));
    ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

    const gaspi_atomic_value_t got = *(gaspi_atomic_value_t *) ((char *) p + off);
    if (old != 0 || got != 5)
    {
      fprintf (stderr, "case 5: fetch_add old %lu, value %lu\n",
               (unsigned long) old, (unsigned long) got);
      bad++;
    }
  }

  //4: [3/2 MB, 7/2 MB) overlaps the inner [1 MB, 2 MB) partly
  if (s2 && use (3, 3 * _1MB / 2, 2 * _1MB, 4))
  {
    transfer (2, 2 * MSG_SIZE, 3, _1MB, 4);
    ASSERT (gaspi_segment_delete (3));
  }

  if (s2)
  {
    ASSERT (gaspi_segment_delete (2));
  }

  if (rank == 0)
  {
    fprintf (stderr, "cases: 5, bad: %d\n", bad);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  free (buf);

  assert (bad == 0);

  return EXIT_SUCCESS;
}
