CFLAGS += -std=c99
ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = autogen.sh

# SOURCES
SRCS = src/GPI2_Env.c src/GPI2_Utility.c src/GPI2_SN.c src/GPI2_Logger.c src/GPI2_Stats.c \
	src/GPI2_Mem.c src/GPI2_Threads.c src/GPI2_Coll.c src/GPI2_IO.c src/GPI2_ATOMIC.c \
	src/GPI2_PASSIVE.c src/GPI2_SEG.c src/GPI2_GRP.c src/GPI2_CONFIG.c src/GPI2_CM.c \
	src/GPI2.c

# GENERAL C FLAGS
lib_gaspi_device_includes = -Iinclude/ -Isrc/devices/include
lib_default_cflags = -Wall -D_GNU_SOURCE
lib_debug_flags = -g -O0
# CFLAGS =

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

# tests/tests/test_utils.o : tests/tests/test_utils.c
# 	$(CC) $(CFLAGS) $(AM_CPPFLAGS) -Itests/tests -c $<
# BUILT_SOURCES = tests/tests/test_utils.o

# FORTRAN MODULES
if WITH_FORTRAN
FMODDIR=include/
src/GASPI.o : src/GASPI.f90
	$(FC) $(FCFLAGS) @ac_cv_fc_module_output_flag@ $(FMODDIR) -O2 -c $< -o $@
src/GASPI_Ext.o : src/GASPI_Ext.f90 src/GASPI.o
	$(FC) $(FCFLAGS) @ac_cv_fc_module_output_flag@ $(FMODDIR) -O2 -c $< -o $@
all-local : src/GASPI.o src/GASPI_Ext.o
endif

# TESTS
# TODO: Call only when checking
tests/tests/test_utils.o : tests/tests/test_utils.c
	$(CC) $(CFLAGS) -Iinclude/ -Itests/tests/ -c $< -o $@

all-local : tests/tests/test_utils.o

# TESTS_ENVIRONMENT = \
# 	buildir=$(top_builddir)/tests

# AM_TESTS_ENVIRONMENT = \
# 	$(CC) $(CFLAGS) -I. -c tests/tests/test_utils.c

## TESTS SUBDIR
### INIT
test_BUILT_TESTS = \
	tests/tests/init/proc_init.bin \
	# tests/tests/init/proc_init_timeout.bin \
	# tests/tests/init/cmd_line_args.bin \
	# tests/tests/init/kill_procs.bin \
	# tests/tests/init/print_to.bin \
	# tests/tests/init/null_ptrs.bin \
	# tests/tests/init/numa_check.bin \
	# tests/tests/init/strong_sym.bin \
	# tests/tests/init/local_rank.bin \
	# tests/tests/init/initialized.bin

check_PROGRAMS = $(test_BUILT_TESTS)
TESTS = $(test_BUILT_TESTS)
LOG_DRIVER = cd tests; ./runtests.sh -m machinefile

LDADD = tests/tests/test_utils.o -lGPI2dbg -lpthread
if WITH_INFINIBAND
LDADD += -L@ac_lib_infiniband@ -libverbs
endif
AM_CFLAGS = $(libGPI2_la_CFLAGS)
AM_CPPFLAGS = -Itests/tests $(libGPI2_la_CPPFLAGS)

check : $(lib_LTLIBRARIES) $(test_BUILT_TESTS)
	cp $(test_BUILT_TESTS) tests/bin


CLEANFILES = *~ *.bak src/*.o src/*.mod include/*.mod \
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
