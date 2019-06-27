ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = autogen.sh

# SOURCES
SRCS = src/GPI2_Env.c src/GPI2_Utility.c src/GPI2_SN.c src/GPI2_Logger.c src/GPI2_Stats.c \
	src/GPI2_Mem.c src/GPI2_Threads.c src/GPI2_Coll.c src/GPI2_IO.c src/GPI2_ATOMIC.c \
	src/GPI2_PASSIVE.c src/GPI2_SEG.c src/GPI2_GRP.c src/GPI2_CONFIG.c src/GPI2_CM.c \
	src/GPI2.c

# HEADERS TO INSTALL
gaspi_headers_dir = src/gaspi/include
include_HEADERS = $(gaspi_headers_dir)/GASPI.h $(gaspi_headers_dir)/GASPI_Ext.h $(gaspi_headers_dir)/GASPI_Threads.h $(gaspi_headers_dir)/PGASPI.h

# GENERAL C FLAGS
lib_gaspi_device_includes = -I$(gaspi_headers_dir) -Isrc/devices/include
lib_default_cflags = -Wall -D_GNU_SOURCE
lib_debug_flags = -g -O0

# MPI MODE
if WITH_MPI
lib_default_cflags += -I@ac_inc_mpi@ -DGPI2_WITH_MPI
endif

# LOADLEVER SUPPORT
if WITH_LOADLEVER
lib_default_cflags += -DLOADLEVER
endif

# DEVICES SPECIFICS
if WITH_INFINIBAND
SRCS += src/devices/ib/GPI2_IB_IO.c src/devices/ib/GPI2_IB_PASSIVE.c src/devices/ib/GPI2_IB_ATOMIC.c \
	src/devices/ib/GPI2_IB_GRP.c src/devices/ib/GPI2_IB_SEG.c src/devices/ib/GPI2_IB.c
lib_gaspi_device_includes += -Isrc/ -Isrc/devices/ib -I@ac_inc_infiniband@ -DGPI2_DEVICE_IB
else
SRCS += src/devices/tcp/list.c src/devices/tcp/rb.c src/devices/tcp/tcp_device.c \
	src/devices/tcp/GPI2_TCP.c src/devices/tcp/GPI2_TCP_IO.c src/devices/tcp/GPI2_TCP_SEG.c \
	src/devices/tcp/GPI2_TCP_PASSIVE.c src/devices/tcp/GPI2_TCP_ATOMIC.c \
	src/devices/tcp/GPI2_TCP_GRP.c
lib_gaspi_device_includes += -Isrc/ -Idevices/tcp -DGPI2_DEVICE_TCP
endif

# MAIN LIB
lib_LTLIBRARIES = libGPI2.la
libGPI2_la_SOURCES = $(SRCS)
libGPI2_la_CFLAGS = $(lib_default_cflags) -O2
libGPI2_la_CPPFLAGS = $(lib_gaspi_device_includes)

# DEBUG AND STAT LIBRARIES
if ENABLE_DEBUG
lib_LTLIBRARIES += libGPI2dbg.la
libGPI2dbg_la_SOURCES = $(SRCS)
libGPI2dbg_la_CFLAGS = $(lib_default_cflags) $(lib_debug_flags) -DDEGUB
libGPI2dbg_la_CPPFLAGS = $(lib_gaspi_device_includes)
endif

if ENABLE_STAT
lib_LTLIBRARIES += libGPI2stat.la
libGPI2stat_la_SOURCES = $(SRCS)
libGPI2stat_la_CFLAGS = $(lib_default_cflags) $(lib_debug_flags) -DDEGUB -DGPI2_STATS
libGPI2stat_la_CPPFLAGS = $(lib_gaspi_device_includes)
endif

# FORTRAN MODULES
if WITH_FORTRAN
FMODDIR=$(gaspi_headers_dir)
$(FMODDIR)/gaspi_types.mod : src/GASPI_types.f90
	$(FC) $(FCFLAGS) @ac_cv_fc_module_output_flag@ $(FMODDIR) -O2 -c $<
$(FMODDIR)/gaspi.mod : src/GASPI.f90 $(FMODDIR)/gaspi_types.mod
	$(FC) $(FCFLAGS) @ac_cv_fc_module_output_flag@ $(FMODDIR) -O2 -c $<
$(FMODDIR)/gaspi_ext.mod : src/GASPI_Ext.f90 $(FMODDIR)/gaspi_types.mod $(FMODDIR)/gaspi.mod
	$(FC) $(FCFLAGS) @ac_cv_fc_module_output_flag@ $(FMODDIR) -O2 -c $<
include_HEADERS += $(FMODDIR)/gaspi_types.mod $(FMODDIR)/gaspi.mod $(FMODDIR)/gaspi_ext.mod
endif

# TESTS
## TESTS SUBDIR
tests_subdir = tests/tests
### INIT
init_tests = \
	tests/bin/proc_init.bin \
	tests/bin/proc_init_timeout.bin \
	tests/bin/cmd_line_args.bin \
	tests/bin/kill_procs.bin \
	tests/bin/print_to.bin \
	tests/bin/null_ptrs.bin \
	tests/bin/numa_check.bin \
	tests/bin/strong_sym.bin \
	tests/bin/local_rank.bin \
	tests/bin/initialized.bin

tests_bin_proc_init_bin_SOURCES = tests/tests/init/proc_init.c $(lib_LTLIBRARIES)
tests_bin_proc_init_timeout_bin_SOURCES = tests/tests/init/proc_init_timeout.c $(lib_LTLIBRARIES)
tests_bin_cmd_line_args_bin_SOURCES = tests/tests/init/cmd_line_args.c $(lib_LTLIBRARIES)
tests_bin_kill_procs_bin_SOURCES = tests/tests/init/kill_procs.c $(lib_LTLIBRARIES)
tests_bin_print_to_bin_SOURCES = tests/tests/init/print_to.c $(lib_LTLIBRARIES)
tests_bin_null_ptrs_bin_SOURCES = tests/tests/init/null_ptrs.c $(lib_LTLIBRARIES)
tests_bin_numa_check_bin_SOURCES = tests/tests/init/numa_check.c $(lib_LTLIBRARIES)
tests_bin_strong_sym_bin_SOURCES = tests/tests/init/strong_sym.c $(lib_LTLIBRARIES)
tests_bin_local_rank_bin_SOURCES = tests/tests/init/local_rank.c $(lib_LTLIBRARIES)
tests_bin_initialized_bin_SOURCES = tests/tests/init/initialized.c $(lib_LTLIBRARIES)

### CONFIG
config_tests = \
	tests/bin/set_config.bin \
	tests/bin/config_netinfo.bin \
	tests/bin/print_config.bin ##Not include before?

tests_bin_set_config_bin_SOURCES = tests/tests/config/set_config.c
tests_bin_config_netinfo_bin_SOURCES = tests/tests/config/config_netinfo.c
tests_bin_print_config_bin_SOURCES = tests/tests/config/print_config.c

LDADD = tests/tests/test_utils.o -lGPI2dbg -lpthread
if WITH_INFINIBAND
LDADD += -L@ac_lib_infiniband@ -libverbs
endif
AM_CFLAGS = $(libGPI2_la_CFLAGS)
AM_CPPFLAGS = -Itests/tests $(libGPI2_la_CPPFLAGS)

noinst_PROGRAMS = $(init_tests) $(config_tests)



# check_PROGRAMS = $(test_BUILT_TESTS)
# TESTS = $(test_BUILT_TESTS)
# LOG_DRIVER = cd tests; ./runtests.sh -m machinefile

# check : $(lib_LTLIBRARIES) $(test_BUILT_TESTS)
# 	cp $(test_BUILT_TESTS) tests/bin

CLEANFILES = *~ *.bak *.mod \
	rm -fr .deps/ \
	$(gaspi_headers_dir)/*.mod \
	tests/tests/*.o


# all: gpi
# #fortran tests

# gpi:
# $(MAKE) -C src gpi

# fortran:
# $(MAKE) -C src fortran

# mic:
# $(MAKE) -C src mic

# tests: gpi
# cd tests && $(MAKE) && cd ..

# docs:
# @if test "$(DOXYGEN)" = ""; then \
# echo "Doxygen not found."; \
# echo "Install doxygen to be able to generate documentation."; \
# echo "Or consult it online at: http://www.gpi-site.com/gpi2/docs/";\
# false; \
# fi
# doxygen Doxyfile

# clean:
# $(MAKE) -C src clean
# $(MAKE) -C tests clean

# .PHONY: all tests docs clean
