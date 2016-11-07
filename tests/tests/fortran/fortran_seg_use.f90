program main
  use gaspi
  use , intrinsic :: ISO_C_BINDING

  integer(gaspi_return_t) :: ret
  integer(gaspi_rank_t) :: rank, nprocs, remoteRank
  integer(gaspi_segment_id_t) :: seg_id
  integer(gaspi_size_t) :: seg_size
  integer(gaspi_notification_t) :: notf, recv_val
  integer(gaspi_notification_id_t) :: notf_id, start_notf, recvd_notf
  integer(gaspi_offset_t) :: localOff, remoteOff
  integer(gaspi_queue_id_t) :: queue
  integer(gaspi_size_t) :: tsize
  integer(gaspi_memory_description_t) :: mdesc
  integer(gaspi_rank_t), allocatable, target :: arr(:)
  type(c_ptr) :: seg_ptr

  !init
  ret = gaspi_proc_init(GASPI_BLOCK)
  call stop_on_error(ret,"gaspi_proc_init")

  ret = gaspi_proc_rank(rank)
  call stop_on_error(ret,"gaspi_proc_rank")
  ret = gaspi_proc_num(nprocs)
  call stop_on_error(ret,"gaspi_proc_num")

  seg_id = 0
  seg_size = 4 * 1024 * 1024

  allocate ( arr( seg_size / sizeof(rank)) )
  mdesc = 0
  seg_id = 0
  ret = gaspi_segment_use(seg_id, c_loc(arr), seg_size, GASPI_GROUP_ALL, GASPI_BLOCK, mdesc)
  call stop_on_error(ret, "gaspi_segment_use")

  ret = gaspi_segment_ptr(seg_id, seg_ptr)
  call stop_on_error(ret, "gaspi_segment_ptr")

  !set data and sync
  arr(:) = (rank)

  ret = gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK)
  call stop_on_error(ret,"gaspi_barrier")

  !communication
  remoteRank = modulo(rank + 1, nprocs)
  localOff = seg_size / 2
  remoteOff = 0
  queue = 0
  tsize = seg_size / 2
  notf_id = rank
  notf = 1
  ret = gaspi_write_notify(seg_id, localOff, remoteRank, &
       & seg_id, remoteOff, tsize, notf_id, notf, queue, GASPI_BLOCK)
  call stop_on_error(ret,"gaspi_write_notify")

  !wait
  ret = gaspi_wait(queue, GASPI_BLOCK)
  call stop_on_error(ret,"gaspi_wait")

  start_notf = modulo(rank + nprocs - 1, nprocs)
  
  ret = gaspi_notify_waitsome(seg_id, start_notf, 1, recvd_notf, GASPI_BLOCK)
  call stop_on_error(ret,"gaspi_notify_waitsome")

  ret = gaspi_notify_reset(seg_id, recvd_notf, recv_val )
  call stop_on_error(ret,"gaspi_notify_reset")

  if(rank .eq. 0) then
     remoteRank = nprocs - 1
  else
     remoteRank = rank - 1
  end if

  !check data
  do i = 1, seg_size/ 2 /sizeof(rank)
     if( arr(i) .ne. remoteRank) then
	write(*,*) "Rank ", rank, " data is wrong", i, arr(i), remoteRank
	call exit(-1)
     end if
  end do

  !term
  ret = gaspi_proc_term(GASPI_BLOCK)
  call stop_on_error(ret,"gaspi_proc_term")

  contains
    subroutine stop_on_error(rval,msg)
      integer(gaspi_return_t), intent(in) :: rval
      character(len=*), intent(in) :: msg
      if(rval .ne. GASPI_SUCCESS) then
	 print *,msg,' error ', rval
	 call exit(-1)
      endif
    end subroutine stop_on_error
end program main
