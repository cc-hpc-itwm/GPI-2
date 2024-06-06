#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

/* This test performs several (ITERATIONS) barriers on different
 * groups where each group is handled by a thread. That is, there as
 * many groups as threads and each threads does the barriers on its
 * group. The idea is to verify the thread-safety of barriers on
 * different groups (instead of a particular group) when executed
 * concurrently. */

#define ITERATIONS 1000

struct workInfo
{
  gaspi_group_t group;
};

struct workInfo* createInfo()
{
  struct workInfo* wi = malloc (sizeof (struct workInfo));
  assert (wi);

  gaspi_rank_t nprocs, myrank;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  gaspi_group_t g;
  ASSERT (gaspi_group_create (&g));

  for (gaspi_rank_t i = 0; i < nprocs; i++)
  {
    ASSERT (gaspi_group_add (g, i));
  }

  ASSERT (gaspi_group_commit (g, GASPI_BLOCK));

  wi->group = g;

  return wi;
}

void* thread_work (void* args)
{
  struct workInfo* wi = (struct workInfo*) args;

  for (int j = 0; j < ITERATIONS; j++)
  {
    ASSERT (gaspi_barrier (wi->group, GASPI_BLOCK));
  }

  for (int j = 0; j < ITERATIONS; j++)
  {
    gaspi_barrier (wi->group, 10);
  }

  return NULL;
}

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  int nthreads = 2;

  struct workInfo* wi[nthreads];
  for (int t = 0; t < nthreads; t++)
  {
    wi[t] = createInfo();
    assert (wi[t]);
  }

  pthread_t *threads = (pthread_t*) malloc (nthreads * sizeof (pthread_t));
  assert (threads);

  for (int t = 0; t < nthreads; t++)
  {
    pthread_create (&threads[t], NULL, thread_work, wi[t]);
  }

  for (int t = 0; t < nthreads; t++)
  {
    pthread_join (threads[t], NULL);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
