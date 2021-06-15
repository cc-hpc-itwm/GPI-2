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
LDADD += @ac_lib_mpi@  @mpi_extra_flags@
endif

# MICROBENCHMARKS
auxMICSRCS = common.c common.h

micro_tests = bin/allreduce.bin \
     bin/barrier.bin \
     bin/init_time.bin \
     bin/init_time_nobuild.bin \
     bin/nb_allreduce.bin \
     bin/nb_barrier.bin \
     bin/ping_pong.bin \
     bin/read_bw.bin \
     bin/read_notify_bw.bin \
     bin/write_bw.bin \
     bin/write_lat.bin \
     bin/write_notify_bw.bin \
     bin/write_notify_lat.bin

bin_allreduce_bin_SOURCES = allreduce.c $(auxMICSRCS)
bin_barrier_bin_SOURCES = barrier.c $(auxMICSRCS)
bin_init_time_bin_SOURCES = init_time.c $(auxMICSRCS)
bin_init_time_nobuild_bin_SOURCES = init_time_nobuild.c $(auxMICSRCS)
bin_nb_allreduce_bin_SOURCES = nb_allreduce.c $(auxMICSRCS)
bin_nb_barrier_bin_SOURCES = nb_barrier.c $(auxMICSRCS)
bin_ping_pong_bin_SOURCES = ping_pong.c $(auxMICSRCS)
bin_read_bw_bin_SOURCES = read_bw.c $(auxMICSRCS)
bin_read_notify_bw_bin_SOURCES = read_notify_bw.c $(auxMICSRCS)
bin_write_bw_bin_SOURCES = write_bw.c $(auxMICSRCS)
bin_write_lat_bin_SOURCES = write_lat.c $(auxMICSRCS)
bin_write_notify_bw_bin_SOURCES = write_notify_bw.c $(auxMICSRCS)
bin_write_notify_lat_bin_SOURCES = write_notify_lat.c $(auxMICSRCS)

micro_pkglibexecdir=$(prefix)/tests/microbenchmarks/bin
micro_pkglibexec_PROGRAMS = $(micro_tests)

CLEANFILES = *~ *.bak *.bin
