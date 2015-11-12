#include <GASPI.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

namespace
{
  template<typename Fun, typename... Args>
    void success_or_die ( const std::string& function_name
			, Fun&& fun
			, Args&&... args
			)
  {
    gaspi_return_t const rc (fun (std::forward<Args> (args)...));

    if (rc != GASPI_SUCCESS)
    {
      throw std::runtime_error
	( "ERROR: " + function_name + " failed: "
	+ gaspi_error_str (rc)
	+ " (" + std::to_string (rc) + ")"
	);
    }
  }

#define SUCCESS_OR_DIE(F, Args...) success_or_die (#F, F, Args)

  class gaspi
  {
  public:
    gaspi()
      : _unused_segment_id (0)
      , _queue (0)
    {
      SUCCESS_OR_DIE (gaspi_proc_init, GASPI_BLOCK);
      SUCCESS_OR_DIE (gaspi_group_commit, GASPI_GROUP_ALL, GASPI_BLOCK);
      SUCCESS_OR_DIE (gaspi_proc_rank, &_iProc);
      SUCCESS_OR_DIE (gaspi_proc_num, &_nProc);
      SUCCESS_OR_DIE (gaspi_notification_num, &_notification_num);
      SUCCESS_OR_DIE (gaspi_queue_num, &_queue_num);
      SUCCESS_OR_DIE (gaspi_queue_size_max, &_queue_size_max);
    }
    ~gaspi()
    {
      SUCCESS_OR_DIE (gaspi_proc_term, GASPI_BLOCK);
    }

    gaspi_rank_t iProc() const
    {
      return _iProc;
    }
    gaspi_rank_t nProc() const
    {
      return _nProc;
    }
    gaspi_number_t notification_num() const
    {
      return _notification_num;
    }

    template<typename T>
      class segment
    {
    public:
      segment (gaspi_segment_id_t id, gaspi_size_t number_of_elements)
	: _id (id)
	, _number_of_elements (number_of_elements)
      {
	SUCCESS_OR_DIE ( gaspi_segment_create
		       , _id
		       , _number_of_elements * sizeof (T)
		       , GASPI_GROUP_ALL
		       , GASPI_BLOCK
		       , GASPI_MEM_UNINITIALIZED
		       );

	SUCCESS_OR_DIE (gaspi_segment_ptr, _id, &_pointer);
      }
      ~segment()
      {
	SUCCESS_OR_DIE (gaspi_segment_delete, _id);
      }

      gaspi_segment_id_t id() const
      {
	return _id;
      }

      T* begin() const
      {
	return static_cast<T*> (_pointer);
      }
      T* end() const
      {
	return begin() + _number_of_elements;
      }
      template<typename Index>
	T const& at (Index i) const
      {
	if (!(i < _number_of_elements))
	{
	  throw std::invalid_argument ("segment.at");
	}

	return *(begin() + i);
      }

    private:
      gaspi_segment_id_t _id;
      gaspi_size_t _number_of_elements;
      gaspi_pointer_t _pointer;
    };

    template<typename T>
      segment<T> segment_create (gaspi_number_t number_of_elements)
    {
      return segment<T> (_unused_segment_id++, number_of_elements);
    }

    gaspi_queue_id_t queue (gaspi_number_t entries)
    {
      gaspi_number_t queue_size;

      SUCCESS_OR_DIE (gaspi_queue_size, _queue, &queue_size);

      if (queue_size + entries > _queue_size_max)
      {
	_queue = (_queue + 1) % _queue_num;

	SUCCESS_OR_DIE (gaspi_wait, _queue, GASPI_BLOCK);
      }

      return _queue;
    }

  private:
    gaspi_segment_id_t _unused_segment_id;
    gaspi_rank_t _iProc;
    gaspi_rank_t _nProc;
    gaspi_number_t _notification_num;
    gaspi_queue_id_t _queue;
    gaspi_number_t _queue_size_max;
    gaspi_number_t _queue_num;
  };

  std::ostream& operator<< (std::ostream& os, gaspi const& gaspi)
  {
    return os << '(' << gaspi.iProc() << '/' << gaspi.nProc() << ')';
  }

  template<typename T>
    class range
  {
  public:
    range (gaspi const& gaspi, T M)
      : _M (M)
      , _P (gaspi.nProc())
      , _begin (begin (gaspi.iProc()))
      , _end (begin (gaspi.iProc() + 1))
    {}

    T begin() const
    {
      return _begin;
    }
    T end() const
    {
      return _end;
    }
    gaspi_rank_t node (T k) const
    {
      return (k * _P) / _M;
    }

  private:
    T const _M;
    gaspi_rank_t const _P;
    T const _begin;
    T const _end;

    T begin (gaspi_rank_t iProc) const
    {
      return (iProc * _M + _P - 1) / _P;
    }
  };

  template<typename T>
    std::ostream& operator<< (std::ostream& os, range<T> const& range)
  {
    return os << '[' << range.begin() << ".." << range.end() << ')';
  }
}

int main()
try
{
  gaspi gaspi;

  auto const segment
    (gaspi.segment_create<gaspi_notification_t> (gaspi.notification_num()));

  std::iota (segment.begin(), segment.end(), 1);

  range<gaspi_notification_id_t>
    const notifications (gaspi, gaspi.notification_num());

  std::cout << gaspi << ": " << notifications << '\n';

  std::thread waiter
    ( [&segment, &gaspi, &notifications]()
      {
	unsigned open (gaspi.notification_num());

	while (open --> 0)
	{
	  gaspi_notification_id_t notified;

	  SUCCESS_OR_DIE ( gaspi_notify_waitsome
			 , segment.id()
			 , 0
			 , gaspi.notification_num()
			 , &notified
			 , GASPI_BLOCK
			 );

	  gaspi_notification_t value;

	  SUCCESS_OR_DIE (gaspi_notify_reset, segment.id(), notified, &value);

	  if (value - 1 != notified)
	  {
	    throw std::logic_error
	      ( "BUMMER: ["  + std::to_string (gaspi.iProc())
	      + "]: wrong value from "
	      + std::to_string (notifications.node (notified))
	      + ": expected " + std::to_string (notified)
	      + " got: " + std::to_string (value)
	      );
	  }
	}
      }
    );

  for ( gaspi_notification_id_t notification_id (notifications.begin())
      ; notification_id != notifications.end()
      ; ++notification_id
      )
  {
    for (gaspi_rank_t other (0); other < gaspi.nProc(); ++other)
    {
      gaspi_rank_t const receiver ((gaspi.iProc() + other) % gaspi.nProc());
      SUCCESS_OR_DIE ( gaspi_write_notify
		     , segment.id()
		     , notification_id * sizeof (gaspi_notification_t)
		     , receiver
		     , segment.id()
		     , notification_id * sizeof (gaspi_notification_t)
		     , sizeof (gaspi_notification_t)
		     , notification_id
		     , segment.at (notification_id)
		     , gaspi.queue (2)
		     , GASPI_BLOCK
		     );
    };
  }

  waiter.join();

  return 0;
}
catch (std::exception const& e)
{
  std::cerr << "Exception: " << e.what() << '\n';

  return 1;
}
catch (...)
{
  std::cerr << "Unkown exception\n";

  return 2;
}
