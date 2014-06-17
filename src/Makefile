include make.inc

SRCS += GPI2_IB_IO.c GPI2_IB_PASSIVE.c GPI2_IB_ATOMIC.c GPI2_IB_GRP.c GPI2_IB.c \
 GPI2_Env.c GPI2_Utility.c GPI2_SN.c GPI2_Logger.c GPI2_Stats.c GPI2_Mem.c GPI2_Threads.c GPI2.c 
HDRS += GPI2_IB.h GPI2_Env.h GPI2_Utility.h GPI_Types.h GPI2_SN.h GPI2.h

OBJS = $(SRCS:.c=.o)
OBJS_DBG = $(SRCS:.c=.dbg.o)
OBJS_MIC = $(SRCS:.c=.mic.o)

OBJ_FORTRAN = GASPI.o

GPI2 = libGPI2.a
GPI2_DBG = libGPI2-dbg.a

gpi: $(GPI2) $(GPI2_DBG)

fortran: $(OBJ_FORTRAN)

$(GPI2): $(OBJS)
	rm -f ../lib64/$(GPI2)
	ar crs ../lib64/$(GPI2) $(OBJS)

$(GPI2_DBG): $(OBJS_DBG) 
	rm -f ../lib64/$(GPI2_DBG)
	ar crs ../lib64/$(GPI2_DBG) $(OBJS_DBG)

$(OBJ_FORTRAN): GASPI.f90
	@if test "$(HAS_FORTRAN)" == "" ; then \
	echo "Fortran compiler (gfortran) not found!" ;\
	false; \
	fi;
	$(FORTRANC) -O2 -c $< 
	cp gaspi.mod gaspi_types.mod ../include;

mic: $(OBJS_MIC)
	rm -f ../lib64/mic/$(GPI2)
	ar crs ../lib64/mic/$(GPI2) $(OBJS_MIC)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

%.dbg.o: %.c
	$(CC) $(DBG_CFLAGS) -DDEBUG $(INCLUDES) -c $<  -o $@

%.mic.o: %.c
	icc -mmic $(MIC_CFLAGS) $(INCLUDES) -DMIC -c $<  -o $@

clean:
	$(RM) *.o *~ ../lib64/$(GPI2) ../lib64/$(GPI2_DBG) ../lib64/mic/$(GPI2)
	$(RM) *.mod ../include/gaspi_types.mod ../include/gaspi.mod

depend: $(SRCS)
	makedepend $(INCLUDES) $^

.PHONY: depend clean

# DO NOT DELETE THIS LINE -- make depend needs it

GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/cuda_runtime_api.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/host_defines.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/builtin_types.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/device_types.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/driver_types.h
GPI2_GPU.o: /usr/include/limits.h /usr/include/features.h
GPI2_GPU.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_GPU.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_GPU.o: /usr/include/limits.h /usr/include/bits/posix1_lim.h
GPI2_GPU.o: /usr/include/bits/local_lim.h /usr/include/linux/limits.h
GPI2_GPU.o: /usr/include/bits/posix2_lim.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/surface_types.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/texture_types.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/vector_types.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/cuda_device_runtime_api.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/cuda.h
GPI2_GPU.o: /usr/include/stdlib.h /usr/include/bits/waitflags.h
GPI2_GPU.o: /usr/include/bits/waitstatus.h /usr/include/endian.h
GPI2_GPU.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
GPI2_GPU.o: /usr/include/sys/types.h /usr/include/bits/types.h
GPI2_GPU.o: /usr/include/bits/typesizes.h /usr/include/time.h
GPI2_GPU.o: /usr/include/sys/select.h /usr/include/bits/select.h
GPI2_GPU.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
GPI2_GPU.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
GPI2_GPU.o: /usr/include/alloca.h /usr/include/string.h
GPI2_GPU.o: /usr/include/xlocale.h ../include/GASPI.h GPI2.h GPI2_Types.h
GPI2_GPU.o: /usr/include/pthread.h /usr/include/sched.h
GPI2_GPU.o: /usr/include/bits/sched.h /usr/include/bits/setjmp.h
GPI2_GPU.o: GPI2_Utility.h /usr/include/errno.h /usr/include/bits/errno.h
GPI2_GPU.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
GPI2_GPU.o: /usr/include/asm-generic/errno.h
GPI2_GPU.o: /usr/include/asm-generic/errno-base.h /usr/include/stdio.h
GPI2_GPU.o: /usr/include/libio.h /usr/include/_G_config.h
GPI2_GPU.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
GPI2_GPU.o: /usr/include/bits/sys_errlist.h GPI2_IB.h
GPI2_GPU.o: /usr/include/infiniband/verbs.h /usr/include/stdint.h
GPI2_GPU.o: /usr/include/bits/wchar.h /usr/include/infiniband/ofa_verbs.h
GPI2_GPU.o: /usr/include/infiniband/verbs_exp.h
GPI2_GPU.o: /usr/include/infiniband/driver.h
GPI2_GPU.o: /usr/include/infiniband/kern-abi.h /usr/include/linux/types.h
GPI2_GPU.o: /usr/include/asm/types.h /usr/include/asm-generic/types.h
GPI2_GPU.o: /usr/include/asm-generic/int-ll64.h
GPI2_GPU.o: /usr/include/asm/bitsperlong.h
GPI2_GPU.o: /usr/include/asm-generic/bitsperlong.h
GPI2_GPU.o: /usr/include/linux/posix_types.h /usr/include/linux/stddef.h
GPI2_GPU.o: /usr/include/asm/posix_types.h /usr/include/asm/posix_types_64.h
GPI2_GPU.o: GPI2_GPU.h /shared/apps/cuda/CUDA-v5.5.22/include/cuda_runtime.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/host_config.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/channel_descriptor.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/driver_functions.h
GPI2_GPU.o: /shared/apps/cuda/CUDA-v5.5.22/include/vector_functions.h
GPI2_GPU.o: ../include/GASPI_GPU.h
GPI2_IB_IO.o: ../include/GASPI.h GPI2.h GPI2_Types.h /usr/include/pthread.h
GPI2_IB_IO.o: /usr/include/features.h /usr/include/sys/cdefs.h
GPI2_IB_IO.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
GPI2_IB_IO.o: /usr/include/gnu/stubs-64.h /usr/include/endian.h
GPI2_IB_IO.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
GPI2_IB_IO.o: /usr/include/sched.h /usr/include/bits/types.h
GPI2_IB_IO.o: /usr/include/bits/typesizes.h /usr/include/time.h
GPI2_IB_IO.o: /usr/include/bits/sched.h /usr/include/bits/pthreadtypes.h
GPI2_IB_IO.o: /usr/include/bits/setjmp.h GPI2_Utility.h /usr/include/errno.h
GPI2_IB_IO.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
GPI2_IB_IO.o: /usr/include/asm/errno.h /usr/include/asm-generic/errno.h
GPI2_IB_IO.o: /usr/include/asm-generic/errno-base.h /usr/include/string.h
GPI2_IB_IO.o: /usr/include/xlocale.h /usr/include/stdlib.h
GPI2_IB_IO.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
GPI2_IB_IO.o: /usr/include/sys/types.h /usr/include/sys/select.h
GPI2_IB_IO.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
GPI2_IB_IO.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
GPI2_IB_IO.o: /usr/include/alloca.h /usr/include/stdio.h /usr/include/libio.h
GPI2_IB_IO.o: /usr/include/_G_config.h /usr/include/wchar.h
GPI2_IB_IO.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
GPI2_IB_IO.o: GPI2_IB.h /usr/include/infiniband/verbs.h /usr/include/stdint.h
GPI2_IB_IO.o: /usr/include/bits/wchar.h /usr/include/infiniband/ofa_verbs.h
GPI2_IB_IO.o: /usr/include/infiniband/verbs_exp.h
GPI2_IB_IO.o: /usr/include/infiniband/driver.h
GPI2_IB_IO.o: /usr/include/infiniband/kern-abi.h /usr/include/linux/types.h
GPI2_IB_IO.o: /usr/include/asm/types.h /usr/include/asm-generic/types.h
GPI2_IB_IO.o: /usr/include/asm-generic/int-ll64.h
GPI2_IB_IO.o: /usr/include/asm/bitsperlong.h
GPI2_IB_IO.o: /usr/include/asm-generic/bitsperlong.h
GPI2_IB_IO.o: /usr/include/linux/posix_types.h /usr/include/linux/stddef.h
GPI2_IB_IO.o: /usr/include/asm/posix_types.h
GPI2_IB_IO.o: /usr/include/asm/posix_types_64.h
GPI2_IB_PASSIVE.o: GPI2.h ../include/GASPI.h GPI2_Types.h
GPI2_IB_PASSIVE.o: /usr/include/pthread.h /usr/include/features.h
GPI2_IB_PASSIVE.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_IB_PASSIVE.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_IB_PASSIVE.o: /usr/include/endian.h /usr/include/bits/endian.h
GPI2_IB_PASSIVE.o: /usr/include/bits/byteswap.h /usr/include/sched.h
GPI2_IB_PASSIVE.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_IB_PASSIVE.o: /usr/include/time.h /usr/include/bits/sched.h
GPI2_IB_PASSIVE.o: /usr/include/bits/pthreadtypes.h
GPI2_IB_PASSIVE.o: /usr/include/bits/setjmp.h GPI2_Utility.h
GPI2_IB_PASSIVE.o: /usr/include/errno.h /usr/include/bits/errno.h
GPI2_IB_PASSIVE.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
GPI2_IB_PASSIVE.o: /usr/include/asm-generic/errno.h
GPI2_IB_PASSIVE.o: /usr/include/asm-generic/errno-base.h
GPI2_IB_PASSIVE.o: /usr/include/string.h /usr/include/xlocale.h
GPI2_IB_PASSIVE.o: /usr/include/stdlib.h /usr/include/bits/waitflags.h
GPI2_IB_PASSIVE.o: /usr/include/bits/waitstatus.h /usr/include/sys/types.h
GPI2_IB_PASSIVE.o: /usr/include/sys/select.h /usr/include/bits/select.h
GPI2_IB_PASSIVE.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
GPI2_IB_PASSIVE.o: /usr/include/sys/sysmacros.h /usr/include/alloca.h
GPI2_IB_PASSIVE.o: /usr/include/stdio.h /usr/include/libio.h
GPI2_IB_PASSIVE.o: /usr/include/_G_config.h /usr/include/wchar.h
GPI2_IB_PASSIVE.o: /usr/include/bits/stdio_lim.h
GPI2_IB_PASSIVE.o: /usr/include/bits/sys_errlist.h GPI2_IB.h
GPI2_IB_PASSIVE.o: /usr/include/infiniband/verbs.h /usr/include/stdint.h
GPI2_IB_PASSIVE.o: /usr/include/bits/wchar.h
GPI2_IB_PASSIVE.o: /usr/include/infiniband/ofa_verbs.h
GPI2_IB_PASSIVE.o: /usr/include/infiniband/verbs_exp.h
GPI2_IB_PASSIVE.o: /usr/include/infiniband/driver.h
GPI2_IB_PASSIVE.o: /usr/include/infiniband/kern-abi.h
GPI2_IB_PASSIVE.o: /usr/include/linux/types.h /usr/include/asm/types.h
GPI2_IB_PASSIVE.o: /usr/include/asm-generic/types.h
GPI2_IB_PASSIVE.o: /usr/include/asm-generic/int-ll64.h
GPI2_IB_PASSIVE.o: /usr/include/asm/bitsperlong.h
GPI2_IB_PASSIVE.o: /usr/include/asm-generic/bitsperlong.h
GPI2_IB_PASSIVE.o: /usr/include/linux/posix_types.h
GPI2_IB_PASSIVE.o: /usr/include/linux/stddef.h /usr/include/asm/posix_types.h
GPI2_IB_PASSIVE.o: /usr/include/asm/posix_types_64.h
GPI2_IB_ATOMIC.o: GPI2.h ../include/GASPI.h GPI2_Types.h
GPI2_IB_ATOMIC.o: /usr/include/pthread.h /usr/include/features.h
GPI2_IB_ATOMIC.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_IB_ATOMIC.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_IB_ATOMIC.o: /usr/include/endian.h /usr/include/bits/endian.h
GPI2_IB_ATOMIC.o: /usr/include/bits/byteswap.h /usr/include/sched.h
GPI2_IB_ATOMIC.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_IB_ATOMIC.o: /usr/include/time.h /usr/include/bits/sched.h
GPI2_IB_ATOMIC.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
GPI2_IB_ATOMIC.o: GPI2_Utility.h /usr/include/errno.h
GPI2_IB_ATOMIC.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
GPI2_IB_ATOMIC.o: /usr/include/asm/errno.h /usr/include/asm-generic/errno.h
GPI2_IB_ATOMIC.o: /usr/include/asm-generic/errno-base.h /usr/include/string.h
GPI2_IB_ATOMIC.o: /usr/include/xlocale.h /usr/include/stdlib.h
GPI2_IB_ATOMIC.o: /usr/include/bits/waitflags.h
GPI2_IB_ATOMIC.o: /usr/include/bits/waitstatus.h /usr/include/sys/types.h
GPI2_IB_ATOMIC.o: /usr/include/sys/select.h /usr/include/bits/select.h
GPI2_IB_ATOMIC.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
GPI2_IB_ATOMIC.o: /usr/include/sys/sysmacros.h /usr/include/alloca.h
GPI2_IB_ATOMIC.o: /usr/include/stdio.h /usr/include/libio.h
GPI2_IB_ATOMIC.o: /usr/include/_G_config.h /usr/include/wchar.h
GPI2_IB_ATOMIC.o: /usr/include/bits/stdio_lim.h
GPI2_IB_ATOMIC.o: /usr/include/bits/sys_errlist.h GPI2_IB.h
GPI2_IB_ATOMIC.o: /usr/include/infiniband/verbs.h /usr/include/stdint.h
GPI2_IB_ATOMIC.o: /usr/include/bits/wchar.h
GPI2_IB_ATOMIC.o: /usr/include/infiniband/ofa_verbs.h
GPI2_IB_ATOMIC.o: /usr/include/infiniband/verbs_exp.h
GPI2_IB_ATOMIC.o: /usr/include/infiniband/driver.h
GPI2_IB_ATOMIC.o: /usr/include/infiniband/kern-abi.h
GPI2_IB_ATOMIC.o: /usr/include/linux/types.h /usr/include/asm/types.h
GPI2_IB_ATOMIC.o: /usr/include/asm-generic/types.h
GPI2_IB_ATOMIC.o: /usr/include/asm-generic/int-ll64.h
GPI2_IB_ATOMIC.o: /usr/include/asm/bitsperlong.h
GPI2_IB_ATOMIC.o: /usr/include/asm-generic/bitsperlong.h
GPI2_IB_ATOMIC.o: /usr/include/linux/posix_types.h
GPI2_IB_ATOMIC.o: /usr/include/linux/stddef.h /usr/include/asm/posix_types.h
GPI2_IB_ATOMIC.o: /usr/include/asm/posix_types_64.h
GPI2_IB_GRP.o: GPI2.h ../include/GASPI.h GPI2_Types.h /usr/include/pthread.h
GPI2_IB_GRP.o: /usr/include/features.h /usr/include/sys/cdefs.h
GPI2_IB_GRP.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
GPI2_IB_GRP.o: /usr/include/gnu/stubs-64.h /usr/include/endian.h
GPI2_IB_GRP.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
GPI2_IB_GRP.o: /usr/include/sched.h /usr/include/bits/types.h
GPI2_IB_GRP.o: /usr/include/bits/typesizes.h /usr/include/time.h
GPI2_IB_GRP.o: /usr/include/bits/sched.h /usr/include/bits/pthreadtypes.h
GPI2_IB_GRP.o: /usr/include/bits/setjmp.h GPI2_Utility.h /usr/include/errno.h
GPI2_IB_GRP.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
GPI2_IB_GRP.o: /usr/include/asm/errno.h /usr/include/asm-generic/errno.h
GPI2_IB_GRP.o: /usr/include/asm-generic/errno-base.h /usr/include/string.h
GPI2_IB_GRP.o: /usr/include/xlocale.h /usr/include/stdlib.h
GPI2_IB_GRP.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
GPI2_IB_GRP.o: /usr/include/sys/types.h /usr/include/sys/select.h
GPI2_IB_GRP.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
GPI2_IB_GRP.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
GPI2_IB_GRP.o: /usr/include/alloca.h /usr/include/stdio.h
GPI2_IB_GRP.o: /usr/include/libio.h /usr/include/_G_config.h
GPI2_IB_GRP.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
GPI2_IB_GRP.o: /usr/include/bits/sys_errlist.h GPI2_IB.h
GPI2_IB_GRP.o: /usr/include/infiniband/verbs.h /usr/include/stdint.h
GPI2_IB_GRP.o: /usr/include/bits/wchar.h /usr/include/infiniband/ofa_verbs.h
GPI2_IB_GRP.o: /usr/include/infiniband/verbs_exp.h
GPI2_IB_GRP.o: /usr/include/infiniband/driver.h
GPI2_IB_GRP.o: /usr/include/infiniband/kern-abi.h /usr/include/linux/types.h
GPI2_IB_GRP.o: /usr/include/asm/types.h /usr/include/asm-generic/types.h
GPI2_IB_GRP.o: /usr/include/asm-generic/int-ll64.h
GPI2_IB_GRP.o: /usr/include/asm/bitsperlong.h
GPI2_IB_GRP.o: /usr/include/asm-generic/bitsperlong.h
GPI2_IB_GRP.o: /usr/include/linux/posix_types.h /usr/include/linux/stddef.h
GPI2_IB_GRP.o: /usr/include/asm/posix_types.h
GPI2_IB_GRP.o: /usr/include/asm/posix_types_64.h
GPI2_IB.o: /usr/include/errno.h /usr/include/features.h
GPI2_IB.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_IB.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_IB.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
GPI2_IB.o: /usr/include/asm/errno.h /usr/include/asm-generic/errno.h
GPI2_IB.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/mman.h
GPI2_IB.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_IB.o: /usr/include/bits/mman.h /usr/include/sys/time.h
GPI2_IB.o: /usr/include/time.h /usr/include/bits/time.h
GPI2_IB.o: /usr/include/sys/select.h /usr/include/bits/select.h
GPI2_IB.o: /usr/include/bits/sigset.h /usr/include/sys/timeb.h
GPI2_IB.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
GPI2_IB.o: /usr/include/bits/environments.h /usr/include/bits/confname.h
GPI2_IB.o: /usr/include/getopt.h ../include/GASPI.h GPI2.h GPI2_Types.h
GPI2_IB.o: /usr/include/pthread.h /usr/include/endian.h
GPI2_IB.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
GPI2_IB.o: /usr/include/sched.h /usr/include/bits/sched.h
GPI2_IB.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
GPI2_IB.o: GPI2_Utility.h /usr/include/string.h /usr/include/xlocale.h
GPI2_IB.o: /usr/include/stdlib.h /usr/include/bits/waitflags.h
GPI2_IB.o: /usr/include/bits/waitstatus.h /usr/include/sys/types.h
GPI2_IB.o: /usr/include/sys/sysmacros.h /usr/include/alloca.h
GPI2_IB.o: /usr/include/stdio.h /usr/include/libio.h /usr/include/_G_config.h
GPI2_IB.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
GPI2_IB.o: /usr/include/bits/sys_errlist.h GPI2_IB.h
GPI2_IB.o: /usr/include/infiniband/verbs.h /usr/include/stdint.h
GPI2_IB.o: /usr/include/bits/wchar.h /usr/include/infiniband/ofa_verbs.h
GPI2_IB.o: /usr/include/infiniband/verbs_exp.h
GPI2_IB.o: /usr/include/infiniband/driver.h
GPI2_IB.o: /usr/include/infiniband/kern-abi.h /usr/include/linux/types.h
GPI2_IB.o: /usr/include/asm/types.h /usr/include/asm-generic/types.h
GPI2_IB.o: /usr/include/asm-generic/int-ll64.h /usr/include/asm/bitsperlong.h
GPI2_IB.o: /usr/include/asm-generic/bitsperlong.h
GPI2_IB.o: /usr/include/linux/posix_types.h /usr/include/linux/stddef.h
GPI2_IB.o: /usr/include/asm/posix_types.h /usr/include/asm/posix_types_64.h
GPI2_IB.o: GPI2_SN.h
GPI2_Env.o: /usr/include/sched.h /usr/include/features.h
GPI2_Env.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_Env.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_Env.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_Env.o: /usr/include/time.h /usr/include/bits/sched.h
GPI2_Env.o: /usr/include/stdio.h /usr/include/libio.h
GPI2_Env.o: /usr/include/_G_config.h /usr/include/wchar.h
GPI2_Env.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
GPI2_Env.o: /usr/include/stdlib.h /usr/include/bits/waitflags.h
GPI2_Env.o: /usr/include/bits/waitstatus.h /usr/include/endian.h
GPI2_Env.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
GPI2_Env.o: /usr/include/sys/types.h /usr/include/sys/select.h
GPI2_Env.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
GPI2_Env.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
GPI2_Env.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
GPI2_Env.o: /usr/include/string.h /usr/include/xlocale.h GPI2_Env.h
GPI2_Env.o: GPI2_Types.h /usr/include/pthread.h /usr/include/bits/setjmp.h
GPI2_Env.o: ../include/GASPI.h GPI2_Utility.h /usr/include/errno.h
GPI2_Env.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
GPI2_Env.o: /usr/include/asm/errno.h /usr/include/asm-generic/errno.h
GPI2_Env.o: /usr/include/asm-generic/errno-base.h
GPI2_Utility.o: /usr/include/errno.h /usr/include/features.h
GPI2_Utility.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_Utility.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_Utility.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
GPI2_Utility.o: /usr/include/asm/errno.h /usr/include/asm-generic/errno.h
GPI2_Utility.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
GPI2_Utility.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_Utility.o: /usr/include/time.h /usr/include/bits/time.h
GPI2_Utility.o: /usr/include/sys/select.h /usr/include/bits/select.h
GPI2_Utility.o: /usr/include/bits/sigset.h GPI2.h ../include/GASPI.h
GPI2_Utility.o: GPI2_Types.h /usr/include/pthread.h /usr/include/endian.h
GPI2_Utility.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
GPI2_Utility.o: /usr/include/sched.h /usr/include/bits/sched.h
GPI2_Utility.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
GPI2_Utility.o: GPI2_Utility.h /usr/include/string.h /usr/include/xlocale.h
GPI2_Utility.o: /usr/include/stdlib.h /usr/include/bits/waitflags.h
GPI2_Utility.o: /usr/include/bits/waitstatus.h /usr/include/sys/types.h
GPI2_Utility.o: /usr/include/sys/sysmacros.h /usr/include/alloca.h
GPI2_Utility.o: /usr/include/stdio.h /usr/include/libio.h
GPI2_Utility.o: /usr/include/_G_config.h /usr/include/wchar.h
GPI2_Utility.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
GPI2_SN.o: /usr/include/errno.h /usr/include/features.h
GPI2_SN.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_SN.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_SN.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
GPI2_SN.o: /usr/include/asm/errno.h /usr/include/asm-generic/errno.h
GPI2_SN.o: /usr/include/asm-generic/errno-base.h /usr/include/fcntl.h
GPI2_SN.o: /usr/include/bits/fcntl.h /usr/include/sys/types.h
GPI2_SN.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_SN.o: /usr/include/time.h /usr/include/endian.h
GPI2_SN.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
GPI2_SN.o: /usr/include/sys/select.h /usr/include/bits/select.h
GPI2_SN.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
GPI2_SN.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
GPI2_SN.o: /usr/include/bits/stat.h /usr/include/netdb.h
GPI2_SN.o: /usr/include/netinet/in.h /usr/include/stdint.h
GPI2_SN.o: /usr/include/bits/wchar.h /usr/include/sys/socket.h
GPI2_SN.o: /usr/include/sys/uio.h /usr/include/bits/uio.h
GPI2_SN.o: /usr/include/bits/socket.h /usr/include/bits/sockaddr.h
GPI2_SN.o: /usr/include/asm/socket.h /usr/include/asm-generic/socket.h
GPI2_SN.o: /usr/include/asm/sockios.h /usr/include/asm-generic/sockios.h
GPI2_SN.o: /usr/include/bits/in.h /usr/include/rpc/netdb.h
GPI2_SN.o: /usr/include/bits/netdb.h /usr/include/netinet/tcp.h
GPI2_SN.o: /usr/include/signal.h /usr/include/bits/signum.h
GPI2_SN.o: /usr/include/bits/siginfo.h /usr/include/bits/sigaction.h
GPI2_SN.o: /usr/include/bits/sigcontext.h /usr/include/bits/sigstack.h
GPI2_SN.o: /usr/include/sys/ucontext.h /usr/include/bits/sigthread.h
GPI2_SN.o: /usr/include/sys/resource.h /usr/include/bits/resource.h
GPI2_SN.o: /usr/include/sys/timeb.h /usr/include/sys/epoll.h
GPI2_SN.o: /usr/include/stdlib.h /usr/include/bits/waitflags.h
GPI2_SN.o: /usr/include/bits/waitstatus.h /usr/include/alloca.h
GPI2_SN.o: /usr/include/string.h /usr/include/xlocale.h /usr/include/unistd.h
GPI2_SN.o: /usr/include/bits/posix_opt.h /usr/include/bits/environments.h
GPI2_SN.o: /usr/include/bits/confname.h /usr/include/getopt.h GPI2_SN.h
GPI2_SN.o: /usr/include/stdio.h /usr/include/libio.h /usr/include/_G_config.h
GPI2_SN.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
GPI2_SN.o: /usr/include/bits/sys_errlist.h ../include/GASPI.h GPI2_Types.h
GPI2_SN.o: /usr/include/pthread.h /usr/include/sched.h
GPI2_SN.o: /usr/include/bits/sched.h /usr/include/bits/setjmp.h GPI2_IB.h
GPI2_SN.o: /usr/include/infiniband/verbs.h
GPI2_SN.o: /usr/include/infiniband/ofa_verbs.h
GPI2_SN.o: /usr/include/infiniband/verbs_exp.h
GPI2_SN.o: /usr/include/infiniband/driver.h
GPI2_SN.o: /usr/include/infiniband/kern-abi.h /usr/include/linux/types.h
GPI2_SN.o: /usr/include/asm/types.h /usr/include/asm-generic/types.h
GPI2_SN.o: /usr/include/asm-generic/int-ll64.h /usr/include/asm/bitsperlong.h
GPI2_SN.o: /usr/include/asm-generic/bitsperlong.h
GPI2_SN.o: /usr/include/linux/posix_types.h /usr/include/linux/stddef.h
GPI2_SN.o: /usr/include/asm/posix_types.h /usr/include/asm/posix_types_64.h
GPI2_Logger.o: /usr/include/fcntl.h /usr/include/features.h
GPI2_Logger.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_Logger.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_Logger.o: /usr/include/bits/fcntl.h /usr/include/sys/types.h
GPI2_Logger.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_Logger.o: /usr/include/time.h /usr/include/endian.h
GPI2_Logger.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
GPI2_Logger.o: /usr/include/sys/select.h /usr/include/bits/select.h
GPI2_Logger.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
GPI2_Logger.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
GPI2_Logger.o: /usr/include/bits/stat.h /usr/include/netdb.h
GPI2_Logger.o: /usr/include/netinet/in.h /usr/include/stdint.h
GPI2_Logger.o: /usr/include/bits/wchar.h /usr/include/sys/socket.h
GPI2_Logger.o: /usr/include/sys/uio.h /usr/include/bits/uio.h
GPI2_Logger.o: /usr/include/bits/socket.h /usr/include/bits/sockaddr.h
GPI2_Logger.o: /usr/include/asm/socket.h /usr/include/asm-generic/socket.h
GPI2_Logger.o: /usr/include/asm/sockios.h /usr/include/asm-generic/sockios.h
GPI2_Logger.o: /usr/include/bits/in.h /usr/include/rpc/netdb.h
GPI2_Logger.o: /usr/include/bits/netdb.h /usr/include/netinet/tcp.h
GPI2_Logger.o: /usr/include/pthread.h /usr/include/sched.h
GPI2_Logger.o: /usr/include/bits/sched.h /usr/include/bits/setjmp.h
GPI2_Logger.o: /usr/include/string.h /usr/include/xlocale.h
GPI2_Logger.o: /usr/include/sys/stat.h /usr/include/stdio.h
GPI2_Logger.o: /usr/include/libio.h /usr/include/_G_config.h
GPI2_Logger.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
GPI2_Logger.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
GPI2_Logger.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
GPI2_Logger.o: /usr/include/alloca.h /usr/include/unistd.h
GPI2_Logger.o: /usr/include/bits/posix_opt.h /usr/include/bits/environments.h
GPI2_Logger.o: /usr/include/bits/confname.h /usr/include/getopt.h GPI2.h
GPI2_Logger.o: ../include/GASPI.h GPI2_Types.h GPI2_Utility.h
GPI2_Logger.o: /usr/include/errno.h /usr/include/bits/errno.h
GPI2_Logger.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
GPI2_Logger.o: /usr/include/asm-generic/errno.h
GPI2_Logger.o: /usr/include/asm-generic/errno-base.h
GPI2_Stats.o: ../include/GASPI.h
GPI2_Mem.o: GPI2_Mem.h /usr/include/stdio.h /usr/include/features.h
GPI2_Mem.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_Mem.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_Mem.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_Mem.o: /usr/include/libio.h /usr/include/_G_config.h
GPI2_Mem.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
GPI2_Mem.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
GPI2_Mem.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
GPI2_Mem.o: /usr/include/endian.h /usr/include/bits/endian.h
GPI2_Mem.o: /usr/include/bits/byteswap.h /usr/include/sys/types.h
GPI2_Mem.o: /usr/include/time.h /usr/include/sys/select.h
GPI2_Mem.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
GPI2_Mem.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
GPI2_Mem.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
GPI2_Mem.o: /usr/include/string.h /usr/include/xlocale.h
GPI2_Mem.o: /usr/include/sys/resource.h /usr/include/bits/resource.h
GPI2_Mem.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
GPI2_Mem.o: /usr/include/bits/environments.h /usr/include/bits/confname.h
GPI2_Mem.o: /usr/include/getopt.h ../include/GASPI.h
GPI2_Threads.o: /usr/include/pthread.h /usr/include/features.h
GPI2_Threads.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
GPI2_Threads.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
GPI2_Threads.o: /usr/include/endian.h /usr/include/bits/endian.h
GPI2_Threads.o: /usr/include/bits/byteswap.h /usr/include/sched.h
GPI2_Threads.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2_Threads.o: /usr/include/time.h /usr/include/bits/sched.h
GPI2_Threads.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
GPI2_Threads.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
GPI2_Threads.o: /usr/include/bits/environments.h /usr/include/bits/confname.h
GPI2_Threads.o: /usr/include/getopt.h ../include/GASPI_Threads.h
GPI2_Threads.o: ../include/GASPI.h GPI2_Utility.h /usr/include/errno.h
GPI2_Threads.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
GPI2_Threads.o: /usr/include/asm/errno.h /usr/include/asm-generic/errno.h
GPI2_Threads.o: /usr/include/asm-generic/errno-base.h /usr/include/string.h
GPI2_Threads.o: /usr/include/xlocale.h /usr/include/stdlib.h
GPI2_Threads.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
GPI2_Threads.o: /usr/include/sys/types.h /usr/include/sys/select.h
GPI2_Threads.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
GPI2_Threads.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
GPI2_Threads.o: /usr/include/alloca.h /usr/include/stdio.h
GPI2_Threads.o: /usr/include/libio.h /usr/include/_G_config.h
GPI2_Threads.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
GPI2_Threads.o: /usr/include/bits/sys_errlist.h
GPI2.o: /usr/include/errno.h /usr/include/features.h /usr/include/sys/cdefs.h
GPI2.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
GPI2.o: /usr/include/gnu/stubs-64.h /usr/include/bits/errno.h
GPI2.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
GPI2.o: /usr/include/asm-generic/errno.h
GPI2.o: /usr/include/asm-generic/errno-base.h /usr/include/pthread.h
GPI2.o: /usr/include/endian.h /usr/include/bits/endian.h
GPI2.o: /usr/include/bits/byteswap.h /usr/include/sched.h
GPI2.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
GPI2.o: /usr/include/time.h /usr/include/bits/sched.h
GPI2.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
GPI2.o: /usr/include/signal.h /usr/include/bits/sigset.h
GPI2.o: /usr/include/bits/signum.h /usr/include/bits/siginfo.h
GPI2.o: /usr/include/bits/sigaction.h /usr/include/bits/sigcontext.h
GPI2.o: /usr/include/bits/sigstack.h /usr/include/sys/ucontext.h
GPI2.o: /usr/include/bits/sigthread.h /usr/include/sys/resource.h
GPI2.o: /usr/include/bits/resource.h /usr/include/bits/time.h
GPI2.o: /usr/include/sys/socket.h /usr/include/sys/uio.h
GPI2.o: /usr/include/sys/types.h /usr/include/sys/select.h
GPI2.o: /usr/include/bits/select.h /usr/include/sys/sysmacros.h
GPI2.o: /usr/include/bits/uio.h /usr/include/bits/socket.h
GPI2.o: /usr/include/bits/sockaddr.h /usr/include/asm/socket.h
GPI2.o: /usr/include/asm-generic/socket.h /usr/include/asm/sockios.h
GPI2.o: /usr/include/asm-generic/sockios.h /usr/include/sys/timeb.h
GPI2.o: /usr/include/sys/utsname.h /usr/include/bits/utsname.h
GPI2.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
GPI2.o: /usr/include/bits/environments.h /usr/include/bits/confname.h
GPI2.o: /usr/include/getopt.h ../include/GASPI.h GPI2.h GPI2_Types.h
GPI2.o: GPI2_Utility.h /usr/include/string.h /usr/include/xlocale.h
GPI2.o: /usr/include/stdlib.h /usr/include/bits/waitflags.h
GPI2.o: /usr/include/bits/waitstatus.h /usr/include/alloca.h
GPI2.o: /usr/include/stdio.h /usr/include/libio.h /usr/include/_G_config.h
GPI2.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
GPI2.o: /usr/include/bits/sys_errlist.h GPI2_Env.h GPI2_IB.h
GPI2.o: /usr/include/infiniband/verbs.h /usr/include/stdint.h
GPI2.o: /usr/include/bits/wchar.h /usr/include/infiniband/ofa_verbs.h
GPI2.o: /usr/include/infiniband/verbs_exp.h /usr/include/infiniband/driver.h
GPI2.o: /usr/include/infiniband/kern-abi.h /usr/include/linux/types.h
GPI2.o: /usr/include/asm/types.h /usr/include/asm-generic/types.h
GPI2.o: /usr/include/asm-generic/int-ll64.h /usr/include/asm/bitsperlong.h
GPI2.o: /usr/include/asm-generic/bitsperlong.h
GPI2.o: /usr/include/linux/posix_types.h /usr/include/linux/stddef.h
GPI2.o: /usr/include/asm/posix_types.h /usr/include/asm/posix_types_64.h
GPI2.o: GPI2_Mem.h GPI2_SN.h
