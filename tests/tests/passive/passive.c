#include <test_utils.h>

int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t P, myrank;
  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  int * int_GlbMem;
  gaspi_pointer_t _vptr;

  ASSERT(gaspi_segment_ptr(0, &_vptr));

  int_GlbMem = ( int *) _vptr;

  const gaspi_size_t msgSize = sizeof(int);

  if(myrank == 0)
    {
      int_GlbMem[0] = 11223344;
      gaspi_rank_t n;
      for(n = 1; n < P; n++)
	{
	  ASSERT(gaspi_passive_send(0, 0, n, msgSize, GASPI_BLOCK));
	}
    }
  else
    {
      gaspi_rank_t sender;
      ASSERT(gaspi_passive_receive(0, 0, &sender, msgSize, GASPI_BLOCK));
      assert( int_GlbMem[0] == 11223344 );
    }
  
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
