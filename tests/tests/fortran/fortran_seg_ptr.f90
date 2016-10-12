program main
  use gaspi
  use , intrinsic :: ISO_C_BINDING

  integer(gaspi_rank_t) :: rank, nprocs
  integer(gaspi_segment_id_t) :: seg_id
  integer(gaspi_size_t) :: seg_size
  integer(gaspi_alloc_t) :: seg_alloc_policy
  integer(gaspi_rank_t), pointer :: arr(:)
  type(c_ptr) :: seg_ptr

  !init
  ret = gaspi_proc_init(GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_init failed"
     call exit(-1)
  end if

  ret = gaspi_proc_rank(rank)
  ret = gaspi_proc_num(nprocs)

  seg_id = 0
  seg_size = 4 * 1024 * 1024 * 1024

  !create segment
  seg_alloc_policy = GASPI_MEM_UNINITIALIZED
  ret = gaspi_segment_create(INT(0,1), seg_size ,GASPI_GROUP_ALL, GASPI_BLOCK, seg_alloc_policy)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_create_segment failed"
     call exit(-1)
  end if

  !get segment pointer
  ret = gaspi_segment_ptr(INT(0,1), seg_ptr)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_segment_ptr failed"
     call exit(-1)
  end if

  !convert c pointer to fortran pointer
  call c_f_pointer(seg_ptr, arr, shape=[seg_size/sizeof(rank)])

  !sync
  ret = gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_barrier failed"
     call exit(-1)
  end if

  !term
  ret = gaspi_proc_term(GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_term failed"
     call exit(-1)
  end if

end program main
