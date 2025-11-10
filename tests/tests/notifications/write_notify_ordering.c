#include <stdio.h>

#include <GASPI_Ext.h>
#include <test_utils.h>

int rank_neighb (int idx, gaspi_rank_t own_rank, gaspi_rank_t num_ranks)
{
  if (idx == 0)
  {
    return own_rank == 0 ? num_ranks - 1 : (own_rank - 1);
  }
  else
  {
    return (own_rank == num_ranks - 1) ? 0 : (own_rank + 1);
  }
}


int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  const int segment_size_int = 65536;

  gaspi_pointer_t segPtrVoid;
  ASSERT (gaspi_segment_create (0,
                                segment_size_int * sizeof (int),
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK,
                                GASPI_ALLOC_DEFAULT));

  ASSERT (gaspi_segment_ptr(0, &segPtrVoid));

  ASSERT (gaspi_segment_create (1,
                                numranks * sizeof (int),
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK,
                                GASPI_ALLOC_DEFAULT));


  volatile int* segPtr_ = (volatile int*)segPtrVoid;
  for (int i=0; i < segment_size_int; i++)
  {
    segPtr_[i] = -1;
  }
  segPtr_[0] = myrank;


  //Test
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for (int i = 0; i < 2; ++i)
  {
    gaspi_write_notify (0, 0, rank_neighb (i, myrank, numranks),
                        0, (i + 1) * sizeof(int), sizeof(int),
                        i, 1,
                        0, GASPI_BLOCK);
  }

  int cnt = 2;
  char error = 0;

  while (cnt > 0)
  {
    gaspi_notification_t old_notification_val = 0;
    for (int i = 0; i < 2; ++i)
    {
      //note: we reset without waitsome
      ASSERT (gaspi_notify_reset (0, i, &old_notification_val));
      if (old_notification_val)
      {
        if (segPtr_[i + 1] != rank_neighb (1 - i, myrank, numranks))
        {
          error = 1;
        }
        --cnt;
      }
      else
      {
        gaspi_delay();
      }
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (error == 1)
  {
    if (segPtr_[1] != rank_neighb (1, myrank, numranks) ||
        segPtr_[2] != rank_neighb (0, myrank, numranks))
    {
      error = 2;
    }
  }

  gaspi_pointer_t segControlPtr;
  ASSERT (gaspi_segment_ptr(1, &segControlPtr));

  volatile int* errors = (volatile int*) segControlPtr;
  errors[0] = error;

  if (myrank == 0)
  {

    for (int r = 1; r < numranks; ++r)
    {
        gaspi_rank_t rem_rank;
        ASSERT (gaspi_passive_receive (1,
                                       r * sizeof (int),
                                       &rem_rank,
                                       sizeof (int),
                                       GASPI_BLOCK));

        assert (errors[r] == 0);
    }

    assert (errors[0] == 0);
  }
  else
  {
    ASSERT (gaspi_passive_send (1,
                                0,
                                0,
                                sizeof(int),
                                GASPI_BLOCK));
  }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));
}
