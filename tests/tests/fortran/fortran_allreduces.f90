module my_reduce

  use gaspi
  implicit none

contains

    function my_reduce_operation(op_one,op_two,op_res, &
&       op_state,num,element_size,timeout) &
&       result ( res ) bind(C,name="my_reduce_operation")

      implicit none

      integer(gaspi_number_t), intent(in), value :: num        
      integer(c_int), intent(out) :: op_one(num)
      integer(c_int), intent(out) :: op_two(num)
      integer(c_int), intent(out) :: op_res(num)
      integer(c_int), intent(out) :: op_state(num)

      integer(gaspi_size_t), value :: element_size
      integer(gaspi_timeout_t), value :: timeout
      integer(gaspi_return_t) :: res
      integer i

      do i = 1, num
         op_res(i) = max(op_one(i),op_two(i))
      enddo

      res = GASPI_SUCCESS

    end function my_reduce_operation

end module my_reduce

program allreduce

  use gaspi
  use my_reduce
  implicit none

  integer(gaspi_size_t) :: sizeof_int
  integer(gaspi_return_t) :: res
  integer(gaspi_rank_t) :: rank, num, i
  integer(c_int), dimension(1), target :: buffer_send
  integer(c_int), dimension(1), target :: buffer_recv
  integer(c_int), dimension(1), target :: reduce_state
  integer(gaspi_number_t) :: num_elem
  integer(gaspi_int) :: operation
  integer(gaspi_int) :: datatyp
  integer(gaspi_group_t) :: group
  integer(gaspi_timeout_t) :: timeout
  type(c_funptr) :: fproc

  sizeof_int = 4
  num_elem = 1
  group = GASPI_GROUP_ALL
  timeout = GASPI_BLOCK
  datatyp = GASPI_TYPE_INT
  operation = GASPI_OP_MAX
  fproc = c_funloc(my_reduce_operation)

  res = gaspi_proc_init(timeout)
  if(res .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_init failed"
     call exit(-1)
  end if

  res = gaspi_barrier(group, timeout)
  if(res .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  res = gaspi_proc_rank(rank)
  if(res .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_rank failed"
     call exit(-1)
  end if

  res = gaspi_proc_num(num)
  if(res .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_num failed"
     call exit(-1)
  end if

! allreduce
  buffer_send(1) = rank
  buffer_recv(1) = -1

  res = gaspi_allreduce(C_LOC(buffer_send),C_LOC(buffer_recv),num_elem, &
&         operation,datatyp,group,timeout)  
  if(res .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_allreduce failed"
     call exit(-1)
  end if

  if (buffer_recv(1) .ne. (num -1)) then
     write (*,*) "Wrong result"
     call exit(-1)
  end if

  ! allreduce_user
  buffer_send(1)  = rank
  buffer_recv(1)  = -1
  reduce_state(1) =  0

  res = gaspi_allreduce_user(C_LOC(buffer_send),C_LOC(buffer_recv),num_elem,sizeof_int, &
&         fproc,C_LOC(reduce_state),group,timeout)
  if(res .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_allreduce_user failed"
     call exit(-1)
  end if

  if (buffer_recv(1) .ne. (num -1)) then
     write (*,*) "Wrong result"
     call exit(-1)
  end if

  res = gaspi_barrier(group, timeout)
  if(res .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  res = gaspi_proc_term(timeout)
  if(res .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_term failed"
     call exit(-1)
  end if

end program allreduce


