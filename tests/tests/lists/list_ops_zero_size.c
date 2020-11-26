#include <test_utils.h>

/* Test for list operations with zero size messages */

struct segs_offsets_list
{
  gaspi_number_t nelems;

  gaspi_segment_id_t* local_segs;
  gaspi_offset_t* local_offs;

  gaspi_segment_id_t* remote_segs;
  gaspi_offset_t* remote_offs;

  gaspi_size_t* sizes;
};

struct segs_offsets_list*
create_comm_list (int nelems)
{
  struct segs_offsets_list* l = malloc (sizeof (struct segs_offsets_list));

  if (l != NULL)
  {
    l->nelems = nelems;

    l->local_segs = malloc (nelems * sizeof (gaspi_segment_id_t));
    l->local_offs = malloc (nelems * sizeof (gaspi_offset_t));

    l->remote_segs = malloc (nelems * sizeof (gaspi_segment_id_t));
    l->remote_offs = malloc (nelems * sizeof (gaspi_offset_t));

    l->sizes = malloc (nelems * sizeof (gaspi_size_t));

    if (l->local_segs == NULL || l->local_offs == NULL ||
        l->remote_segs == NULL || l->remote_segs == NULL ||
        l->sizes == NULL)
    {
      free (l->local_segs);
      free (l->local_offs);
      free (l->remote_segs);
      free (l->remote_offs);
      free (l->sizes);
      free (l);

      return NULL;
    }
  }

  return l;
}

void
comm_list_with_n_elems_with_nonzero_size (struct segs_offsets_list* l,
                                          gaspi_number_t n,
                                          gaspi_number_t* elems)
{
  gaspi_number_t counter = 0;

  for (gaspi_number_t n = 0; n < l->nelems; n++)
  {
    if (n == elems[counter])
    {
      l->sizes[n] = sizeof (gaspi_rank_t);
    }
    else
    {
      l->sizes[n] = 0;
    }

    l->local_segs[n] = 0;
    l->local_offs[n] = counter * sizeof (gaspi_rank_t);

    l->remote_segs[n] = 0;
    l->remote_offs[n] =
      (l->nelems * sizeof (gaspi_rank_t)) + (counter * sizeof (gaspi_rank_t));

    if (n == elems[counter])
    {
      counter++;
    }
  }

  assert (counter == n);
}

void
comm_list_with_one_elem_with_nonzero_size (struct segs_offsets_list* l,
                                           gaspi_number_t elem)
{
  gaspi_number_t n = elem;

  return comm_list_with_n_elems_with_nonzero_size (l, 1, &n);
}

void
initialize_data (gaspi_segment_id_t const seg_id, gaspi_number_t nelems)
{
  gaspi_pointer_t _vptr;

  ASSERT (gaspi_segment_ptr (seg_id, &_vptr));

  gaspi_rank_t *mem = (gaspi_rank_t *) _vptr;

  gaspi_rank_t rank;
  ASSERT (gaspi_proc_rank (&rank));

  for (gaspi_rank_t i = 0; i < 2 * nelems; i++)
  {
    mem[i] = rank;
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
}

void
assert_elems_are_set (gaspi_segment_id_t const seg_id,
                      int value_a_begin, int value_a_end, int value_a,
                      int value_b_begin, int value_b_end, int value_b)
{
  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr (seg_id, &_vptr));

  gaspi_rank_t *mem = (gaspi_rank_t *) _vptr;

  for (int i = value_a_begin; i < value_a_end; i++)
  {
    assert (mem[i] == value_a);
  }

  for (int i = value_b_begin; i < value_b_end; i++)
  {
    assert (mem[i] == value_b);
  }

}

void
assert_n_read_elem_are_set (gaspi_segment_id_t seg_id,
                            int nelems_total, int nelems_set)
{
  gaspi_rank_t rank, nprocs;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  gaspi_rank_t rank2read = (rank + 1) % nprocs;

  assert_elems_are_set (seg_id,
                        0, nelems_set, rank2read,
                        nelems_set, nelems_total, rank);
}

void
assert_n_write_elem_are_set (gaspi_segment_id_t seg_id,
                             int nelems_total, int nelems_set)
{
  gaspi_rank_t rank, nprocs;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  gaspi_rank_t rank2check = (rank + nprocs + 1) % nprocs;

  assert_elems_are_set (seg_id,
                        0, nelems_total / 2, rank,
                        nelems_total / 2, nelems_total / 2 + nelems_set,
                        rank2check);
}

void
assert_single_read_elem_is_set (gaspi_segment_id_t seg_id, int nelems)
{
  return assert_n_read_elem_are_set (seg_id, nelems, 1);
}

void
assert_single_write_elem_is_set (gaspi_segment_id_t seg_id, int nelems)
{
  return assert_n_write_elem_are_set (seg_id, nelems, 1);
}

gaspi_rank_t
next_rank()
{
  gaspi_rank_t rank, nprocs;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  return (rank + 1) % nprocs;
}

gaspi_rank_t
previous_rank()
{
  gaspi_rank_t rank, nprocs;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  return (rank + nprocs + 1) % nprocs;
}

gaspi_rank_t
my_rank()
{
  gaspi_rank_t rank;
  ASSERT (gaspi_proc_rank (&rank));

  return rank;
}

void
wait_notification (gaspi_segment_id_t s,
                   gaspi_notification_id_t n)
{
  gaspi_notification_id_t id;
  ASSERT (gaspi_notify_waitsome (s, n, 1, &id, GASPI_BLOCK));

  gaspi_notification_t val;
  ASSERT (gaspi_notify_reset (s, id, &val));
}

void
do_read_list (struct segs_offsets_list* l)
{
  ASSERT (gaspi_read_list (l->nelems,
                           l->local_segs, l->local_offs,
                           next_rank(),
                           l->remote_segs, l->remote_offs, l->sizes,
                           0, GASPI_BLOCK));

  ASSERT (gaspi_wait (0, GASPI_BLOCK));
}

void
do_read_notify_list_wait_notification (struct segs_offsets_list* l)
{
  gaspi_rank_t myrank;
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_read_list_notify (l->nelems,
                                  l->local_segs, l->local_offs,
                                  next_rank(),
                                  l->remote_segs, l->remote_offs, l->sizes,
                                  l->local_segs[0],
                                  myrank,
                                  0, GASPI_BLOCK));

  wait_notification (l->local_segs[0], myrank);

  ASSERT (gaspi_wait (0, GASPI_BLOCK));
}

void
do_write_list_wait_notification (struct segs_offsets_list* l)
{
  gaspi_rank_t myrank;
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_write_list (l->nelems,
                            l->local_segs, l->local_offs,
                            next_rank(),
                            l->remote_segs, l->remote_offs, l->sizes,
                            0, GASPI_BLOCK));

  ASSERT (gaspi_notify (l->local_segs[0],
                        next_rank(),
                        myrank, 1,
                        0, GASPI_BLOCK));

  wait_notification (l->local_segs[0], previous_rank());

  ASSERT (gaspi_wait (0, GASPI_BLOCK));
}

void
do_write_notify_list_wait_notification (struct segs_offsets_list* l)
{
  gaspi_rank_t myrank;
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_write_list_notify (l->nelems,
                                   l->local_segs, l->local_offs, next_rank(),
                                   l->remote_segs, l->remote_offs, l->sizes,
                                   l->local_segs[0], myrank, 1,
                                   0, GASPI_BLOCK));


  wait_notification (l->local_segs[0], previous_rank());

  ASSERT (gaspi_wait (0, GASPI_BLOCK));
}


#define SINGLE_ELEM_TEST_PROLOGUE(SEG, NELEMS, ELEM)                  \
  initialize_data (SEG, NELEMS);                                      \
  struct segs_offsets_list* list = create_comm_list (NELEMS);         \
  comm_list_with_one_elem_with_nonzero_size (list, ELEM)

#define SINGLE_ELEM_READ_TEST_EPILOGUE(NELEMS)          \
  assert_single_read_elem_is_set (0, 2 * NELEMS)

#define SINGLE_ELEM_WRITE_TEST_EPILOGUE(NELEMS)         \
  assert_single_write_elem_is_set (0, 2 * NELEMS)

gaspi_return_t
read_all_zero_size_except_first_element (gaspi_segment_id_t const seg_id,
                                         gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, 0);

  do_read_list (list);

  SINGLE_ELEM_READ_TEST_EPILOGUE(nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
read_notify_all_zero_size_except_first_element (gaspi_segment_id_t const seg_id,
                                                gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, 0);

  do_read_notify_list_wait_notification (list);

  SINGLE_ELEM_READ_TEST_EPILOGUE(nelems);

  return GASPI_SUCCESS;
}


gaspi_return_t
write_all_zero_size_except_first_element (gaspi_segment_id_t const seg_id,
                                          gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, 0);

  do_write_list_wait_notification (list);

  SINGLE_ELEM_WRITE_TEST_EPILOGUE(nelems);

  return GASPI_SUCCESS;
}


gaspi_return_t
write_notify_all_zero_size_except_first_element (gaspi_segment_id_t const seg_id,
                                                 gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, 0);

  do_write_notify_list_wait_notification (list);

  SINGLE_ELEM_WRITE_TEST_EPILOGUE(nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
read_all_zero_size_except_last_element (gaspi_segment_id_t const seg_id,
                                        gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, nelems - 1);

  do_read_list (list);

  SINGLE_ELEM_READ_TEST_EPILOGUE(nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
read_notify_all_zero_size_except_last_element (gaspi_segment_id_t const seg_id,
                                               gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, nelems - 1);

  do_read_notify_list_wait_notification (list);

  SINGLE_ELEM_READ_TEST_EPILOGUE (nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
write_all_zero_size_except_last_element (gaspi_segment_id_t const seg_id,
                                         gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, nelems - 1);

  do_write_list_wait_notification (list);

  SINGLE_ELEM_WRITE_TEST_EPILOGUE (nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
write_notify_all_zero_size_except_last_element (gaspi_segment_id_t const seg_id,
                                                gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, nelems - 1);

  do_write_notify_list_wait_notification (list);

  SINGLE_ELEM_WRITE_TEST_EPILOGUE (nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
read_all_zero_size_except_middle_element (gaspi_segment_id_t const seg_id,
                                          gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, nelems / 2);

  do_read_list (list);

  SINGLE_ELEM_READ_TEST_EPILOGUE (nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
read_notify_all_zero_size_except_middle_element (gaspi_segment_id_t const seg_id,
                                                 gaspi_number_t const nelems)
{
  SINGLE_ELEM_TEST_PROLOGUE (seg_id, nelems, nelems / 2);

  do_read_notify_list_wait_notification (list);

  SINGLE_ELEM_READ_TEST_EPILOGUE (nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
write_all_zero_size_except_middle_element (gaspi_segment_id_t const seg_id,
                                           gaspi_number_t const nelems)
{
  initialize_data (seg_id, nelems);

  struct segs_offsets_list* list = create_comm_list (nelems);

  comm_list_with_one_elem_with_nonzero_size (list, nelems / 2);

  do_write_list_wait_notification (list);

  assert_single_write_elem_is_set (0, 2 * nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
write_notify_all_zero_size_except_middle_element (gaspi_segment_id_t const seg_id,
                                                  gaspi_number_t const nelems)
{
  initialize_data (seg_id, nelems);

  struct segs_offsets_list* list = create_comm_list (nelems);

  comm_list_with_one_elem_with_nonzero_size (list, nelems / 2);

  do_write_notify_list_wait_notification (list);

  assert_single_write_elem_is_set (0, 2 * nelems);

  return GASPI_SUCCESS;
}

gaspi_return_t
read_even_are_zero_size_others_not (gaspi_segment_id_t const seg_id,
                                    gaspi_number_t const nelems)
{
  initialize_data (seg_id, nelems);

  gaspi_number_t odds_num = ((nelems - 1) % 2 != 0 ? 1 : 0) + (nelems / 2);

  gaspi_number_t* list_of_odds = malloc (odds_num * sizeof (gaspi_number_t));
  if (list_of_odds == NULL)
  {
    return GASPI_ERROR;
  }

  gaspi_number_t odd_i = 1;
  for (gaspi_number_t i = 0; i < odds_num; i++)
  {
    list_of_odds[i] = odd_i;
    odd_i += 2;
  }

  struct segs_offsets_list* list = create_comm_list (nelems);

  comm_list_with_n_elems_with_nonzero_size (list, odds_num, list_of_odds);

  do_read_list (list);

  assert_n_read_elem_are_set (0, 2 * nelems, odds_num);

  return GASPI_SUCCESS;
}

gaspi_return_t
read_notify_even_are_zero_size_others_not (gaspi_segment_id_t const seg_id,
                                           gaspi_number_t const nelems)
{
  initialize_data (seg_id, nelems);

  gaspi_number_t odds_num = ((nelems - 1) % 2 != 0 ? 1 : 0) + (nelems / 2);

  gaspi_number_t* list_of_odds = malloc (odds_num * sizeof (gaspi_number_t));
  if (list_of_odds == NULL)
  {
    return GASPI_ERROR;
  }

  gaspi_number_t odd_i = 1;
  for (gaspi_number_t i = 0; i < odds_num; i++)
  {
    list_of_odds[i] = odd_i;
    odd_i += 2;
  }

  struct segs_offsets_list* list = create_comm_list (nelems);

  comm_list_with_n_elems_with_nonzero_size (list, odds_num, list_of_odds);

  do_read_notify_list_wait_notification (list);

  assert_n_read_elem_are_set (0, 2 * nelems, odds_num);

  return GASPI_SUCCESS;
}

gaspi_return_t
write_even_are_zero_size_others_not (gaspi_segment_id_t const seg_id,
                                     gaspi_number_t const nelems)
{
  initialize_data (seg_id, nelems);

  gaspi_number_t odds_num = ((nelems - 1) % 2 != 0 ? 1 : 0) + (nelems / 2);

  gaspi_number_t* list_of_odds = malloc (odds_num * sizeof (gaspi_number_t));
  if (list_of_odds == NULL)
  {
    return GASPI_ERROR;
  }

  gaspi_number_t odd_i = 1;
  for (gaspi_number_t i = 0; i < odds_num; i++)
  {
    list_of_odds[i] = odd_i;
    odd_i += 2;
  }

  struct segs_offsets_list* list = create_comm_list (nelems);

  comm_list_with_n_elems_with_nonzero_size (list, odds_num, list_of_odds);

  do_write_list_wait_notification (list);

  assert_n_write_elem_are_set (0, 2 * nelems, odds_num);

  return GASPI_SUCCESS;
}

gaspi_return_t
write_notify_even_are_zero_size_others_not (gaspi_segment_id_t const seg_id,
                                            gaspi_number_t const nelems)
{
  initialize_data (seg_id, nelems);

  gaspi_number_t odds_num = ((nelems - 1) % 2 != 0 ? 1 : 0) + (nelems / 2);

  gaspi_number_t* list_of_odds = malloc (odds_num * sizeof (gaspi_number_t));
  if (list_of_odds == NULL)
  {
    return GASPI_ERROR;
  }

  gaspi_number_t odd_i = 1;
  for (gaspi_number_t i = 0; i < odds_num; i++)
  {
    list_of_odds[i] = odd_i;
    odd_i += 2;
  }

  struct segs_offsets_list* list = create_comm_list (nelems);

  comm_list_with_n_elems_with_nonzero_size (list, odds_num, list_of_odds);

  do_write_notify_list_wait_notification (list);

  assert_n_write_elem_are_set (0, 2 * nelems, odds_num);

  return GASPI_SUCCESS;
}

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  const gaspi_number_t nelems = 255;

  const gaspi_segment_id_t seg_id = 0;
  ASSERT (gaspi_segment_create
          (seg_id,
           2 * nelems * sizeof (gaspi_rank_t),
           GASPI_GROUP_ALL,
           GASPI_BLOCK,
           GASPI_MEM_INITIALIZED));


  { //read_list
    ASSERT (read_all_zero_size_except_first_element (seg_id, nelems));

    ASSERT (read_all_zero_size_except_last_element (seg_id, nelems));

    ASSERT (read_all_zero_size_except_middle_element (seg_id, nelems));

    ASSERT (read_even_are_zero_size_others_not (seg_id, nelems));
  }

  { //write_list
    ASSERT (write_all_zero_size_except_first_element (seg_id, nelems));

    ASSERT (write_all_zero_size_except_last_element (seg_id, nelems));

    ASSERT (write_all_zero_size_except_middle_element (seg_id, nelems));

    ASSERT (write_even_are_zero_size_others_not (seg_id, nelems));
  }

  { //write_notify_list
    ASSERT (write_notify_all_zero_size_except_first_element (seg_id, nelems));

    ASSERT (write_notify_all_zero_size_except_last_element (seg_id, nelems));

    ASSERT (write_notify_all_zero_size_except_middle_element (seg_id, nelems));

    ASSERT (write_notify_even_are_zero_size_others_not (seg_id, nelems));
  }

  { //read_notify_list
    ASSERT (read_notify_all_zero_size_except_first_element (seg_id, nelems));

    ASSERT (read_notify_all_zero_size_except_last_element (seg_id, nelems));

    ASSERT (read_notify_all_zero_size_except_middle_element (seg_id, nelems));

    ASSERT (read_notify_even_are_zero_size_others_not (seg_id, nelems));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));


  return EXIT_SUCCESS;
}
