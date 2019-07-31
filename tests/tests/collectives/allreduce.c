#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>


int typeSize[] = {
  sizeof (int),
  sizeof (unsigned int),
  sizeof (float),
  sizeof (double),
  sizeof (long),
  sizeof (uint64_t)
};

#define INIT_FUN(type) init_##type
#define INIT_CALL(type, v, n, r) INIT_FUN(type)((v), (n), ((r)))
#define INIT_DECL(type) int INIT_FUN(type)(type *v, gaspi_number_t n, gaspi_rank_t r)
#define INIT_IMPLEM INIT_DECL(INIT_TYPE) {          \
    gaspi_number_t i;                               \
    for(i = 0; i < n; i++)                          \
      v[i] = (INIT_TYPE) r;                         \
    return 1;                                       \
  }                                                 \

#define INIT_TYPE int
INIT_IMPLEM
#undef INIT_TYPE
#define INIT_TYPE uint32_t
  INIT_IMPLEM
#undef INIT_TYPE
#define INIT_TYPE float
  INIT_IMPLEM
#undef INIT_TYPE
#define INIT_TYPE  double
  INIT_IMPLEM
#undef INIT_TYPE
#define INIT_TYPE long
  INIT_IMPLEM
#undef INIT_TYPE
#define INIT_TYPE uint64_t
  INIT_IMPLEM
#undef INIT_TYPE
#define CHECK_FUN(type) check_##type
#define CHECK_CALL(type, v, n, expected) CHECK_FUN(type)((v), (n), (expected))
#define CHECK_DECL(type) int CHECK_FUN(type)(type *v, gaspi_number_t n, gaspi_operation_t op)
#define CHECK_IMPLEM CHECK_DECL(CHECK_TYPE) {                     \
    gaspi_number_t i;                                             \
    CHECK_TYPE expected;                                          \
    gaspi_rank_t myrank, nprocs;                                  \
    ASSERT(gaspi_proc_rank(&myrank));                             \
    ASSERT(gaspi_proc_num(&nprocs));                              \
    switch(op)                                                    \
    {                                                             \
      case GASPI_OP_MIN: expected = 0;                            \
        break;                                                    \
      case GASPI_OP_MAX: expected  = (CHECK_TYPE) nprocs - 1;     \
        break;                                                    \
      case GASPI_OP_SUM:                                          \
        expected = (nprocs * (nprocs -1)) / 2;                    \
        break;                                                    \
    }                                                             \
    for(i = 0; i < n; i++)                                        \
      if(v[i] != expected)                                        \
      {                                                           \
        gaspi_printf("expected %ld %ld\n", expected, v[i]);       \
        return 0;                                                 \
      }                                                           \
    return 1;                                                     \
  }                                                               \

#define CHECK_TYPE int
  CHECK_IMPLEM
#undef CHECK_TYPE
#define CHECK_TYPE uint32_t
  CHECK_IMPLEM
#undef CHECK_TYPE
#define CHECK_TYPE float
  CHECK_IMPLEM
#undef CHECK_TYPE
#define CHECK_TYPE  double
  CHECK_IMPLEM
#undef CHECK_TYPE
#define CHECK_TYPE long
  CHECK_IMPLEM
#undef CHECK_TYPE
#define CHECK_TYPE uint64_t
  CHECK_IMPLEM
  gaspi_return_t testOP (gaspi_operation_t op, gaspi_datatype_t type,
                         gaspi_number_t elems, gaspi_group_t group)
{
  void *send_bf = malloc (elems * typeSize[type]);

  if (send_bf == NULL)
    return GASPI_ERROR;

  void *recv_bf = malloc (elems * typeSize[type]);

  if (recv_bf == NULL)
  {
    free (send_bf);
    return GASPI_ERROR;
  }

  gaspi_rank_t myrank, nprocs;

  ASSERT (gaspi_proc_rank (&myrank));
  ASSERT (gaspi_proc_num (&nprocs));

  //init data
  switch (type)
  {
    case GASPI_TYPE_INT:
      INIT_CALL (int, send_bf, elems, myrank);

      break;
    case GASPI_TYPE_UINT:
      INIT_CALL (uint32_t, send_bf, elems, myrank);
      break;
    case GASPI_TYPE_FLOAT:
      INIT_CALL (float, send_bf, elems, myrank);

      break;
    case GASPI_TYPE_DOUBLE:
      INIT_CALL (double, send_bf, elems, myrank);

      break;
    case GASPI_TYPE_LONG:
      INIT_CALL (long, send_bf, elems, myrank);

      break;
    case GASPI_TYPE_ULONG:
      INIT_CALL (uint64_t, send_bf, elems, myrank);
      break;
  }

  ASSERT (gaspi_barrier (group, GASPI_BLOCK));

  ASSERT (gaspi_allreduce
          (send_bf, recv_bf, elems, op, type, group, GASPI_BLOCK));

  //check data

  int ret;

  switch (type)
  {
    case GASPI_TYPE_INT:
      ret = CHECK_CALL (int, recv_bf, elems, op);

      break;
    case GASPI_TYPE_UINT:
      ret = CHECK_CALL (uint32_t, recv_bf, elems, op);
      break;
    case GASPI_TYPE_FLOAT:
      ret = CHECK_CALL (float, recv_bf, elems, op);

      break;
    case GASPI_TYPE_DOUBLE:
      ret = CHECK_CALL (double, recv_bf, elems, op);

      break;
    case GASPI_TYPE_LONG:
      ret = CHECK_CALL (long, recv_bf, elems, op);

      break;
    case GASPI_TYPE_ULONG:
      ret = CHECK_CALL (uint64_t, recv_bf, elems, op);
      break;
  }

  free (send_bf);
  free (recv_bf);

  if (ret)
    return GASPI_SUCCESS;
  else
    return GASPI_ERROR;
}

int
main (int argc, char *argv[])
{
  gaspi_rank_t nprocs, myrank;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  int n;

  for (n = 1; n <= 255; n++)
  {
    gaspi_datatype_t type;

    for (type = GASPI_TYPE_INT; type <= GASPI_TYPE_ULONG; type++)
    {
      gaspi_operation_t op;

      for (op = GASPI_OP_MIN; op <= GASPI_OP_SUM; op++)
        ASSERT (testOP (op, type, n, GASPI_GROUP_ALL));
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
