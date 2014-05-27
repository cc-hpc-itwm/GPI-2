!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Fortran test that checks different variants of printf!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program main
  use gaspi
  use , intrinsic :: ISO_C_BINDING

  integer(gaspi_return_t) :: ret
  character(len = 128) :: c

  ret = gaspi_proc_init(GASPI_BLOCK)
  if(ret .ne. GASPI_SUCCESS) then
     write(*,*) "gaspi_proc_init failed"
     call exit(-1)
  end if


  write(c,'(a)') "test_string"

  call gaspi_printf(trim(adjustl(c))//C_NEW_LINE//C_NULL_CHAR)

  call gaspi_printf("test_string"//C_NEW_LINE//C_NULL_CHAR)

  call gaspi_printf(trim(adjustl("test_string"))//C_NEW_LINE//C_NULL_CHAR)
  call gaspi_printf("Finish!"//C_NEW_LINE//C_NULL_CHAR)

end program main
