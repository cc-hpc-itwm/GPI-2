#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <test_utils.h>

#include <string>
#include <cassert>
#include <cstring>
#include <stdexcept>

/* DESCRIPTION: Tests gaspi_segment_use using a shmem buffer */
/* STEPS: */
/* - We create a shmem buffer */
/* - Use created buffer as segment (id 0) */
/* - Create a segment as usual (id 1) */
/* - Fill in application buffer with rank value */
/* - write, notify and wait for notification (with right neighbour) */
/* - check that received data equals left neighbour */
/* - clean-up */

namespace
{
  struct shm_opened
  {
    shm_opened (std::string const& path)
      : _fd (::shm_open (path.c_str(), O_RDWR | O_CREAT, S_IRWXU))
    {
      if (_fd < 0)
	{
	  std::string const error (strerror (errno));

	  throw std::runtime_error
	    ("Could not shm_open (" + path + "): " + error);
	}
    }
    ~shm_opened()
    {
      ::close (_fd);
    }
    operator int() const
    {
      return _fd;
    }
  private:
    int const _fd;
  };

  int truncated (int fd, off_t length)
  {
    if (::ftruncate (fd, length) == -1)
      {
	std::string const error (strerror (errno));

	throw std::runtime_error ("Could not truncate:" + error);
      }

    return fd;
  }

  struct mmapped
  {
    mmapped (int fd, size_t size, int prot)
      : _pointer (::mmap (0, size, prot, MAP_SHARED, fd, 0))
    {
      if (_pointer == MAP_FAILED)
	{
	  std::string const error (strerror (errno));

	  throw std::runtime_error ("mmap failed: " + error);
	}
    }
    template<typename T>
    operator T* const() const
    {
      return static_cast<T*> (_pointer);
    }
  private:
    void* const _pointer;
  };

  template<typename T>
  T* shared_memory (std::string const& path, off_t size)
  {
    assert (size >= 0);

    return mmapped ( truncated (shm_opened (path), size)
		     , static_cast<size_t> (size)
		     , PROT_READ | PROT_WRITE
		     );
  }
}

int main(int argc, char *argv[])
{
  gaspi_rank_t rank, nprocs;
  gaspi_notification_id_t id;
  const int num_elems = 1024;

  TSUITE_INIT( argc, argv );

  ASSERT( gaspi_proc_init( GASPI_BLOCK ) );

  ASSERT( gaspi_proc_num( &nprocs ) );
  ASSERT( gaspi_proc_rank( &rank ) );

  const gaspi_rank_t left = (rank + nprocs - 1 ) % nprocs;
  const gaspi_rank_t right = (rank + nprocs + 1) % nprocs;

  std::string const path ("1mib");
  off_t const size (1 << 20);

  void* memory(shared_memory<void>(path, size));

  gaspi_segment_id_t const shmem_seg_id (0);
  gaspi_segment_id_t const usual_seg_id (1);
  gaspi_memory_description_t const memory_description (0);

  ASSERT( gaspi_segment_use ( shmem_seg_id, memory, size,
			      GASPI_GROUP_ALL, GASPI_BLOCK,
			      memory_description
			      ) );

  ASSERT( gaspi_segment_create( usual_seg_id , num_elems * sizeof(int),
				GASPI_GROUP_ALL, GASPI_BLOCK,
				GASPI_MEM_INITIALIZED) );

  /* Get pointer and fill in lower part of segment with data to write */
  gaspi_pointer_t seg_ptr;
  ASSERT( gaspi_segment_ptr( shmem_seg_id, &seg_ptr ) );

  int * buf = (int *) seg_ptr;
  assert( buf != NULL);

  for (int i = 0; i < num_elems; i++)
    {
      buf[i] = rank;
    }

  ASSERT( gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK ) );

  /* write data from shmem segment to neighbour's usual segment */
  ASSERT( gaspi_write(shmem_seg_id, 0, right,
		      usual_seg_id, 0, num_elems * sizeof(int),
		      0, GASPI_BLOCK) );

  ASSERT( gaspi_notify( usual_seg_id, right, 0, 1, 0, GASPI_BLOCK ) );
  ASSERT( gaspi_notify_waitsome( usual_seg_id, 0, 1, &id, GASPI_BLOCK ) );
  ASSERT( gaspi_wait( 0, GASPI_BLOCK ) );

  /* Check data */
  gaspi_pointer_t seg1_ptr;
  ASSERT( gaspi_segment_ptr( usual_seg_id, &seg1_ptr ) );

  int * const recv_buf = (int *) seg1_ptr;
  for (int i = 0; i < num_elems; i++)
    {
      assert(recv_buf[i] == left);
    }

  ASSERT( gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK ) );

  ASSERT ( gaspi_proc_term(GASPI_BLOCK ) );

  return EXIT_SUCCESS;
}
