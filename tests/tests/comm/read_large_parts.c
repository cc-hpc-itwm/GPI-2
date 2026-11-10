#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <test_utils.h>

/* Large reads, which the DC transport splits into parts: sizes and offsets
   that do not divide evenly, around the sizes where the number of parts
   changes, and a queue full of large reads, which must hold as many as
   with small requests. Every byte is checked. */

#define SRC_SIZE (_8MB)
#define DST_SIZE (64 * _1MB)

static unsigned char
pattern (const gaspi_rank_t rank, const gaspi_size_t off)
{
  return (unsigned char) (rank * 131 + off * 7 + (off >> 12));
}

static int
check (const unsigned char *const dst, const gaspi_rank_t from,
       const gaspi_offset_t rem_off, const gaspi_size_t size)
{
  for (gaspi_size_t b = 0; b < size; b++)
  {
    if (dst[b] != pattern (from, rem_off + b))
    {
      gaspi_printf ("read of %lu bytes from %u: byte %lu is %u, not %u\n",
                    size, from, b, dst[b], pattern (from, rem_off + b));
      return -1;
    }
  }

  return 0;
}

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num (&numranks));
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_segment_create
          (0, SRC_SIZE, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
  ASSERT (gaspi_segment_create
          (1, DST_SIZE, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  gaspi_pointer_t src_ptr, dst_ptr;

  ASSERT (gaspi_segment_ptr (0, &src_ptr));
  ASSERT (gaspi_segment_ptr (1, &dst_ptr));

  unsigned char *const src = (unsigned char *) src_ptr;
  unsigned char *const dst = (unsigned char *) dst_ptr;

  for (gaspi_size_t b = 0; b < SRC_SIZE; b++)
  {
    src[b] = pattern (myrank, b);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  const gaspi_size_t sizes[] = {
    16 * 1024 - 1, 16 * 1024, 16 * 1024 + 1, 32 * 1024 + 1,
    48 * 1024 + 3, 64 * 1024 + 5, 128 * 1024 + 1, _1MB + 7, _4MB - 5
  };
  const gaspi_offset_t loc_off = 3, rem_off = 5;

  for (unsigned s = 0; s < sizeof (sizes) / sizeof (sizes[0]); s++)
  {
    for (gaspi_rank_t r = 0; r < numranks; r++)
    {
      memset (dst, 0, sizes[s] + loc_off);

      ASSERT (gaspi_read (1, loc_off, r, 0, rem_off, sizes[s], 0,
                          GASPI_BLOCK));
      ASSERT (gaspi_wait (0, GASPI_BLOCK));

      assert (check (dst + loc_off, r, rem_off, sizes[s]) == 0);
    }
  }

  /* A queue holds queue_size_max large reads, as small ones */
  gaspi_number_t qmax, qsize;

  ASSERT (gaspi_queue_size_max (&qmax));

  const gaspi_rank_t r = (myrank + 1) % numranks;
  const gaspi_size_t large = 48 * 1024 + 3;
  const gaspi_size_t stride = 48 * 1024 + 64;

  assert (qmax * stride <= DST_SIZE);
  memset (dst, 0, DST_SIZE);

  for (gaspi_number_t i = 0; i < qmax; i++)
  {
    ASSERT (gaspi_read (1, i * stride, r, 0, i * 8 + 1, large, 0,
                        GASPI_BLOCK));
  }

  ASSERT (gaspi_queue_size (0, &qsize));
  assert (qsize == qmax);

  EXPECT_FAIL_WITH (gaspi_read (1, 0, r, 0, 0, large, 0, GASPI_BLOCK),
                    GASPI_QUEUE_FULL);

  ASSERT (gaspi_wait (0, GASPI_BLOCK));

  ASSERT (gaspi_queue_size (0, &qsize));
  assert (qsize == 0);

  for (gaspi_number_t i = 0; i < qmax; i++)
  {
    assert (check (dst + i * stride, r, i * 8 + 1, large) == 0);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
