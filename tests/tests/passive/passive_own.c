#include <GASPI_Threads.h>
#include <test_utils.h>

#define RECV_OFF 1024

/* Test passive communiction to own rank with extra receiving thread
   USING GASPI_TEST in recv*/
void*
recvThread(void * arg)
{
  gaspi_rank_t sender;
  gaspi_offset_t recvPos = (RECV_OFF / sizeof(int));
  int * memArray = (int *) arg;

  gaspi_return_t ret = GASPI_ERROR;
  int tid;
  
  ASSERT (gaspi_threads_register(&tid));
  do
    {
      ret = gaspi_passive_receive(0, RECV_OFF, &sender, sizeof(int), GASPI_TEST);
      assert (ret != GASPI_ERROR);
      sleep(1);
    }
  while(ret != GASPI_SUCCESS);

  assert(memArray[recvPos] == 11223344);

  gaspi_threads_sync();

  return NULL;
}

int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_threads_init_user(2));

  int tid;
  ASSERT (gaspi_threads_register(&tid));

  gaspi_rank_t P, myrank;
  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  int * int_GlbMem;
  gaspi_pointer_t _vptr;

  ASSERT(gaspi_segment_ptr(0, &_vptr));

  int_GlbMem = ( int *) _vptr;

  ASSERT(gaspi_threads_run(recvThread, int_GlbMem));

  int_GlbMem[0] = 11223344;

  ASSERT(gaspi_passive_send(0, 0, myrank, sizeof(int), GASPI_BLOCK));

  gaspi_threads_sync();
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
