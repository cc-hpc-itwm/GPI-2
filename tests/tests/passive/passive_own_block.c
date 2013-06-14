#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <MCTP1.h>
#include <test_utils.h>

#define _4GB 4294967296

#define RECV_OFF 1024

void * recvThread(void * arg)
{
  int tid = mctpRegisterThread();
  gaspi_rank_t sender;
  gaspi_offset_t recvPos = (RECV_OFF / sizeof(int));
  int * memArray = (int *) arg;

  gaspi_return_t ret = GASPI_ERROR;

  do
    {
      ret = gaspi_passive_receive(0, RECV_OFF, &sender, sizeof(int), GASPI_BLOCK);
      assert (ret != GASPI_ERROR);
      sleep(1);
    }
  while(ret != GASPI_SUCCESS);

  gaspi_printf("Received msg from %d\n", sender);
  if( memArray[recvPos] != 11223344 )
    gaspi_printf("Wrong value!\n");

  return NULL;
}

int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  mctpInitUser(2);
  int tid = mctpRegisterThread();

  gaspi_rank_t P, myrank;

  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _4GB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  int * int_GlbMem;
  gaspi_pointer_t _vptr;

  ASSERT(gaspi_segment_ptr(0, &_vptr));

  int_GlbMem = ( int *) _vptr;

  mctpStartSingleThread(recvThread, int_GlbMem);

  int_GlbMem[0] = 11223344;

  //send to myself (need trick to avoid blocking)
  gaspi_return_t ret = GASPI_ERROR;
  do
    {
      ret = gaspi_passive_send(0, 0, myrank, sizeof(int), GASPI_BLOCK);
      assert (ret != GASPI_ERROR);
      sleep(1);
    }
  while (ret != GASPI_SUCCESS);

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
