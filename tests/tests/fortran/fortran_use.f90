!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Fortran test that uses *most* of GPI-2 functionality!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program main
  use gaspi
  use gaspi_ext
  use , intrinsic :: ISO_C_BINDING

  integer(gaspi_size_t) :: transfer, bufsize
  integer(gaspi_number_t) :: num
  integer(gaspi_return_t) :: ret
  integer(gaspi_queue_id_t) :: q
  integer(gaspi_rank_t) :: rank, nprocs, remoteRank, local_rank, local_num
  character(LEN=100) :: out_str
  character(LEN=32) :: istr
  integer(gaspi_segment_id_t) :: seg_id
  integer(gaspi_size_t) :: seg_size, arr_sizem, my_seg_size
  integer(gaspi_alloc_t) :: seg_alloc
  integer(gaspi_notification_t) :: notf, recv_val
  integer(gaspi_notification_id_t) :: notf_id, start, recv_notf
  integer(gaspi_offset_t) :: localOff, remoteOff
  integer(gaspi_queue_id_t) :: queue
  integer(gaspi_size_t) :: tsize
  integer(gaspi_rank_t), pointer :: arr(:)
  integer(gaspi_segment_id_t), target :: list_segs(255)
  integer(gaspi_offset_t), target :: list_locoffs(255)
  integer(gaspi_offset_t), target :: list_remoffs(255)
  integer(gaspi_size_t), target :: list_sizes(255)
  integer(gaspi_atomic_value_t) :: atom_value
  integer(gaspi_number_t) :: initialized
  type(c_ptr) :: segs_ptr, locoff_ptr, remoff_ptr, sizes_ptr
  type(c_ptr) :: seg_ptr
  integer :: seg_int_size, recvmsgs, i

  !init
  ret = gaspi_proc_init(GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_init failed"
     call exit(-1)
  end if

  ret = gaspi_initialized(initialized)
  write(*,*) "Is initialized(after): ", initialized

  !capabilities
  ret = gaspi_transfer_size_min(transfer)
  write(*,*) "Min transfer: ", transfer

  ret = gaspi_transfer_size_max(transfer)
  write(*,*) "Max transfer: ", transfer

  ret = gaspi_notification_num(num)
  write(*,*) "Num notifications: ", num

  ret = gaspi_allreduce_buf_size(bufsize)
  write(*,*) "Allreduce buf size: ", bufsize

  ret = gaspi_allreduce_elem_max(num)
  write(*,*) "Allreduce elem max: ", num

  ret = gaspi_rw_list_elem_max(num)
  write(*,*) "List elem max: ", num

  ret = gaspi_queue_size_max(num)
  write(*,*) "Queue size elem max: ", num

  ret = gaspi_queue_num(num)
  write(*,*) "Queue num: ", num

  q = 0
  ret = gaspi_queue_size(q, num)
  write(*,*) "Queue size: ", num

  ret = gaspi_initialized(initialized)
  write(*,*) "Is initialized(before): ", initialized

  ret = gaspi_proc_rank(rank)
  ret = gaspi_proc_num(nprocs)

  ret = gaspi_proc_local_rank(local_rank)
  ret = gaspi_proc_local_num(local_num)

  write(*,*) "Rank :", rank," of ", nprocs, " Local: ", local_rank," of ", local_num

  seg_id = 0
  seg_size = 2 * 1024 * 1024
  seg_int_size = 2 * 1024 * 1024

  seg_alloc = GASPI_MEM_UNINITIALIZED
  ret = gaspi_segment_create(INT(0,1), seg_size, GASPI_GROUP_ALL, GASPI_BLOCK, seg_alloc) !GASPI_MEM_INITIALIZED)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_create_segment failed"
     call exit(-1)
  end if

  ret = gaspi_segment_size(INT(0,1), rank, my_seg_size)

  ret = gaspi_segment_ptr(INT(0,1), seg_ptr)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_segment_ptr failed"
     call exit(-1)
  end if

  call c_f_pointer(seg_ptr, arr, shape=[seg_size/sizeof(rank)])

  arr_size = seg_size/sizeof(rank)

  !set data
  arr(:) = (rank)

  !barrier
  ret = gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  !communication
  ! write to right neighboor
  remoteRank = modulo(rank + 1, nprocs)
  localOff = (rank * sizeof(rank))
  remoteOff = (rank * sizeof(rank))
  queue = 0
  tsize = sizeof(rank)

  !write at the beginning of segment
  ret = gaspi_write(seg_id, localOff, remoteRank, &
       & seg_id, remoteOff, tsize, queue, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "first gaspi_write failed"
     call exit(-1)
  end if

  localOff = (seg_size / 2 ) + (rank * sizeof(rank))
  remoteOff = localOff

  !write in the middle of segment
  ret = gaspi_write(seg_id, localOff, remoteRank, &
       & seg_id, remoteOff, tsize, queue, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_write failed"
     call exit(-1)
  end if

  ret = gaspi_wait(queue, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_wait failed"
     call exit(-1)
  end if

  !notify partner rank
  notf = 70000
  start = 45678
  recv_notf = 0

  notf_id = 45678
  ret = gaspi_notify(seg_id, remoteRank, notf_id, notf, queue, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_notify failed"
     call exit(-1)
  end if

  ret = gaspi_notify_waitsome(seg_id, start, 1, recv_notf, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_notify_waitsome failed"
     call exit(-1)
  end if

  ret = gaspi_notify_reset(seg_id, recv_notf, recv_val )
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_notify_reset failed"
     call exit(-1)
  end if

  !barrier
  ret = gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  if(rank .eq. 0) then
     remoteRank = nprocs - 1
  else
     remoteRank = rank - 1
  end if

  !check data
  i = remoteRank+1
  if( arr(i) .ne. remoteRank) then
     write(*,*) "Rank ", rank, " :data is wrong at ", i, " is ", arr(i), " should be ", remoteRank
     call exit(-1)
  end if

  i = seg_size / 2 / sizeof(rank) + remoteRank + 1
  if( arr(i) .ne. remoteRank) then
     write(*,*) "Rank ", rank, " : middle data is wrong at ", i, " is ", arr(i), " should be ", remoteRank
     call exit(-1)
  end if

  !passive
  localOff = 0
  tsize = 8
  if(rank .eq. 0) then
     do recvmsgs = 1, nprocs - 1
	ret = gaspi_passive_receive(seg_id, localOff, remoteRank, tsize, GASPI_BLOCK)
	if(ret .ne. GASPI_SUCCESS) then
	   write(*,*) "gaspi_passive_receive failed"
	   call exit(-1)
	end if
	if (arr(1) .ne. 42) then
	   write(*, *) "Wrong magic number"
	end if
     end do
  else
     remoteRank = 0
     arr(1) = 42
     ret = gaspi_passive_send(seg_id, localOff, remoteRank, tsize, GASPI_BLOCK)
     if(ret .ne. GASPI_SUCCESS) then
	write(*,*) "gaspi_passive_send failed"
	call exit(-1)
     end if

  endif

  ret = gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  !lists
  arr(:) = (rank)
  num = 255
  tsize = sizeof(rank)
  remoteRank = modulo(rank + 1, nprocs)
  do i = 1, num
     list_segs(i) = 0
     list_locoffs(i) = (255*tsize) + ((i-1)*tsize)
     list_remoffs(i) = (i-1)*tsize
     list_sizes(i) = tsize
  end do

  segs_ptr = c_loc(list_segs(1))
  locoff_ptr = c_loc(list_locoffs(1))
  remoff_ptr = c_loc(list_remoffs(1))
  sizes_ptr = c_loc(list_sizes(1))

  ret = gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  ret = gaspi_read_list(num, segs_ptr, locoff_ptr, remoteRank, segs_ptr, remoff_ptr, sizes_ptr, queue, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_read_list failed"
     call exit(-1)
  end if

  ret = gaspi_wait(queue, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_wait failed"
     call exit(-1)
  end if

  do i = 1, num
     if( arr(i+255) .ne. remoteRank) then
	write(istr,'(i4)')  arr(i)
	out_str = 'wrong data '//trim(istr)//C_NEW_LINE//C_NULL_CHAR
	call gaspi_printf(out_str)
     end if
  end do
  ret = gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  !atomic
  arr(:) = 0
  localOff = 0
  remoteRank = 0
  ret = gaspi_atomic_fetch_add(seg_id, localOff, remoteRank, INT(1,8), atom_value, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_atomic_fetch_add failed"
     call exit(-1)
  end if

  ret = gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  if(rank .eq. 0) then
     ret = gaspi_atomic_fetch_add(seg_id, localOff, remoteRank, INT(0,8), atom_value, GASPI_BLOCK)
     if(ret .ne. GASPI_SUCCESS) then
	write(*,*) "gaspi_atomic_fetch_add failed"
	call exit(-1)
     end if
     if(atom_value .ne. nprocs) then
	write(*,*) "Incorrect atomic value: ", atom_value
     end if
  end if

  !term
  ret = gaspi_proc_term(GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_term failed"
     call exit(-1)
  end if

end program main
