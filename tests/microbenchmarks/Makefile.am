AM_CFLAGS = -g
AM_CPPFLAGS = -I$(abs_top_srcdir)/src/include

LDADD = $(abs_top_builddir)/src/.libs/libGPI2.la

LDFLAGS = -pthread
if WITH_INFINIBAND
LDFLAGS += @ac_lib_infiniband@
endif

if WITH_MPI
AM_CFLAGS += -DGPI2_WITH_MPI
AM_CFLAGS += @ac_inc_mpi@
LDFLAGS += @ac_lib_mpi@  @mpi_extra_flags@
endif

# MICROBENCHMARKS
auxMICSRCS = common.c common.h utils.h

micro_tests = allreduce.bin \
     barrier.bin \
     init_time.bin \
     init_time_nobuild.bin \
     nb_allreduce.bin \
     nb_barrier.bin \
     ping_pong.bin \
     read_bw.bin \
     read_notify_bw.bin \
     write_bw.bin \
     write_lat.bin \
     write_notify_bw.bin \
     write_notify_lat.bin

allreduce_bin_SOURCES = allreduce.c $(auxMICSRCS)
barrier_bin_SOURCES = barrier.c $(auxMICSRCS)
init_time_bin_SOURCES = init_time.c $(auxMICSRCS)
init_time_nobuild_bin_SOURCES = init_time_nobuild.c $(auxMICSRCS)
nb_allreduce_bin_SOURCES = nb_allreduce.c $(auxMICSRCS)
nb_barrier_bin_SOURCES = nb_barrier.c $(auxMICSRCS)
ping_pong_bin_SOURCES = ping_pong.c $(auxMICSRCS)
read_bw_bin_SOURCES = read_bw.c $(auxMICSRCS)
read_notify_bw_bin_SOURCES = read_notify_bw.c $(auxMICSRCS)
write_bw_bin_SOURCES = write_bw.c $(auxMICSRCS)
write_lat_bin_SOURCES = write_lat.c $(auxMICSRCS)
write_notify_bw_bin_SOURCES = write_notify_bw.c $(auxMICSRCS)
write_notify_lat_bin_SOURCES = write_notify_lat.c $(auxMICSRCS)

noinst_PROGRAMS = $(micro_tests)

CLEANFILES = *~ *.bak *.bin
