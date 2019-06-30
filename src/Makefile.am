# SOURCES
SRCS = GPI2_Env.c GPI2_Utility.c GPI2_SN.c GPI2_Logger.c GPI2_Stats.c \
	GPI2_Mem.c GPI2_Threads.c GPI2_Coll.c GPI2_IO.c GPI2_ATOMIC.c \
	GPI2_PASSIVE.c GPI2_SEG.c GPI2_GRP.c GPI2_CONFIG.c GPI2_CM.c \
	GPI2.c
SRCS += GASPI_Ext.f90 GASPI.f90 GASPI_types.f90
SRCS += GPI2_CM.h GPI2_Coll.h GPI2_Env.h GPI2_GRP.h \
	GPI2.h GPI2_Mem.h GPI2_SEG.h GPI2_SN.h GPI2_Stats.h \
	GPI2_Types.h GPI2_Utility.h

# DEVICES SPECIFICS
SRCS += devices/include/GPI2_Dev.h \
	devices/ib/GPI2_IB.h \
	devices/tcp/GPI2_TCP.h devices/tcp/list.h devices/tcp/rb.h devices/tcp/tcp_device.h

if WITH_INFINIBAND
SRCS += devices/ib/GPI2_IB_IO.c devices/ib/GPI2_IB_PASSIVE.c devices/ib/GPI2_IB_ATOMIC.c \
	devices/ib/GPI2_IB_GRP.c devices/ib/GPI2_IB_SEG.c devices/ib/GPI2_IB.c
else
SRCS += devices/tcp/list.c devices/tcp/rb.c devices/tcp/tcp_device.c \
	devices/tcp/GPI2_TCP.c devices/tcp/GPI2_TCP_IO.c devices/tcp/GPI2_TCP_SEG.c \
	devices/tcp/GPI2_TCP_PASSIVE.c devices/tcp/GPI2_TCP_ATOMIC.c \
	devices/tcp/GPI2_TCP_GRP.c
endif

# HEADERS TO INSTALL
include_HEADERS = gaspi/include/GASPI.h gaspi/include/GASPI_Ext.h gaspi/include/GASPI_Threads.h gaspi/include/PGASPI.h

# GENERAL C FLAGS
gaspi_headers_dir = $(abs_top_builddir)/src/gaspi/include
lib_gaspi_device_includes = -I$(gaspi_headers_dir) -I$(abs_top_builddir)/src/devices/include
lib_default_cflags = -Wall -D_GNU_SOURCE
lib_debug_flags = -g -O0

# DEVICES SPECIFICS
if WITH_INFINIBAND
lib_gaspi_device_includes += -I. -I$(abs_top_builddir)/src/devices/ib -I@ac_inc_infiniband@ -DGPI2_DEVICE_IB
else
lib_gaspi_device_includes += -I. -I$(abs_top_builddir)/src/devices/tcp -DGPI2_DEVICE_TCP
endif

# MPI MODE
if WITH_MPI
lib_default_cflags += -I@ac_inc_mpi@ -DGPI2_WITH_MPI
endif

# LOADLEVER SUPPORT
if WITH_LOADLEVER
lib_default_cflags += -DLOADLEVER
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
FMODDIR=gaspi/include
$(FMODDIR)/gaspi_types.mod : GASPI_types.f90
	$(FC) $(FCFLAGS) @ac_cv_fc_module_output_flag@ $(FMODDIR) -O2 -c $< -o GASPI_types.o
$(FMODDIR)/gaspi.mod : GASPI.f90 $(FMODDIR)/gaspi_types.mod
	$(FC) $(FCFLAGS) @ac_cv_fc_module_output_flag@ $(FMODDIR) -O2 -c $< -o GASPI.o
$(FMODDIR)/gaspi_ext.mod : GASPI_Ext.f90 $(FMODDIR)/gaspi_types.mod $(FMODDIR)/gaspi.mod
	$(FC) $(FCFLAGS) @ac_cv_fc_module_output_flag@ $(FMODDIR) -O2 -c $< -o GASPI_Ext.o
nodist_include_HEADERS = $(FMODDIR)/gaspi_types.mod $(FMODDIR)/gaspi.mod $(FMODDIR)/gaspi_ext.mod
endif

# FLAGS FOR TESTING
tests_flags :
	$(shell echo "AM_CFLAGS = $(libGPI2dbg_la_CFLAGS)" > ../tests/tests/compile_rules.in)
	$(shell echo "AM_CPPFLAGS = $(libGPI2dbg_la_CPPFLAGS)" >> ../tests/tests/compile_rules.in)

microbenchmarks_flags :
	$(shell echo "AM_CFLAGS = $(libGPI2_la_CFLAGS)" > ../tests/microbenchmarks/compile_rules.in)
	$(shell echo "AM_CPPFLAGS = $(libGPI2_la_CPPFLAGS)" >> ../tests/microbenchmarks/compile_rules.in)

all-local : tests_flags microbenchmarks_flags

.PHONY : test_flags microbenchmarks_flags

# CLEAN
CLEANFILES = *~ *.bak $(gaspi_headers_dir)/*.mod
