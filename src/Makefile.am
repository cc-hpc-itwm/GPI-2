# SOURCES
SRCS = GPI2_Env.c GPI2_Utility.c GPI2_SN.c GPI2_Logger.c GPI2_Stats.c \
	GPI2_Mem.c GPI2_Threads.c GPI2_Coll.c GPI2_IO.c GPI2_ATOMIC.c \
	GPI2_PASSIVE.c GPI2_SEG.c GPI2_GRP.c GPI2_CONFIG.c GPI2_CM.c \
	GPI2.c
SRCS += GPI2_CM.h GPI2_Coll.h GPI2_Env.h GPI2_GRP.h \
	GPI2.h GPI2_Mem.h GPI2_SEG.h GPI2_SN.h GPI2_Stats.h \
	GPI2_Types.h GPI2_Utility.h

# DEVICES GENERIC
SRCS += devices/include/GPI2_Dev.h

# HEADERS TO INSTALL
include_HEADERS = include/GASPI.h include/GASPI_Ext.h include/GASPI_Threads.h \
	include/PGASPI.h

# GENERAL C FLAGS
default_includes = -I$(abs_top_srcdir)/src -I$(abs_top_srcdir)/src/include -I$(abs_top_srcdir)/src/devices/include
lib_default_cflags = -Wall -D_GNU_SOURCE

# DEVICES SPECIFICS
if WITH_INFINIBAND
SRCS += devices/ib/GPI2_IB_IO.c devices/ib/GPI2_IB_PASSIVE.c devices/ib/GPI2_IB_ATOMIC.c \
devices/ib/GPI2_IB_GRP.c devices/ib/GPI2_IB_SEG.c devices/ib/GPI2_IB.c
SRCS += devices/ib/GPI2_IB.h
default_includes += -I$(abs_top_srcdir)/src/devices/ib @ac_inc_infiniband@ -DGPI2_DEVICE_IB
if WITH_INFINIBAND_EXT
default_includes += -DGPI2_EXP_VERBS
endif
else
SRCS += devices/tcp/list.c devices/tcp/rb.c devices/tcp/tcp_device.c \
devices/tcp/GPI2_TCP.c devices/tcp/GPI2_TCP_IO.c devices/tcp/GPI2_TCP_SEG.c \
devices/tcp/GPI2_TCP_PASSIVE.c devices/tcp/GPI2_TCP_ATOMIC.c \
devices/tcp/GPI2_TCP_GRP.c
SRCS += devices/tcp/GPI2_TCP.h devices/tcp/list.h devices/tcp/rb.h devices/tcp/tcp_device.h
default_includes += -I$(abs_top_srcdir)/src/devices/tcp -DGPI2_DEVICE_TCP
endif

# MPI MODE
if WITH_MPI
lib_default_cflags += -DGPI2_WITH_MPI
lib_default_cflags += @ac_inc_mpi@
endif

# LOADLEVER SUPPORT
if WITH_LOADLEVELER
lib_default_cflags += -DLOADLEVELER
endif

# MAIN LIB
lib_LTLIBRARIES = libGPI2.la
libGPI2_la_SOURCES = $(SRCS)
libGPI2_la_CFLAGS = $(lib_default_cflags) -O2
libGPI2_la_CPPFLAGS = $(default_includes)

# DEBUG AND STAT LIBRARIES
if ENABLE_DEBUG
lib_LTLIBRARIES += libGPI2-dbg.la
libGPI2_dbg_la_SOURCES = $(SRCS)
libGPI2_dbg_la_CFLAGS = $(lib_default_cflags) -g -O0 -DDEBUG
libGPI2_dbg_la_CPPFLAGS = $(default_includes)
endif

if ENABLE_STAT
lib_LTLIBRARIES += libGPI2-stats.la
libGPI2_stats_la_SOURCES = $(SRCS)
libGPI2_stats_la_CFLAGS = $(lib_default_cflags) -g -O0 -DGPI2_STATS -DDEBUG
libGPI2_stats_la_CPPFLAGS = $(default_includes)
endif

# FORTRAN MODULES
EXTRA_DIST = GASPI_types.f90 GASPI.f90 GASPI_Ext.f90
if WITH_FORTRAN
gaspi_types.mod: GASPI_types.f90
	$(FC) $(FCFLAGS) -O2 -c $< -o GASPI_types.o
gaspi.mod: GASPI.f90 gaspi_types.mod
	$(FC) $(FCFLAGS) -O2 -c $< -o GASPI.o
gaspi_ext.mod: GASPI_Ext.f90 gaspi_types.mod gaspi.mod
	$(FC) $(FCFLAGS) -O2 -c $< -o GASPI_Ext.o
nodist_include_HEADERS = gaspi_types.mod gaspi.mod gaspi_ext.mod
endif

# PKG-CONFIG
EXTRA_DIST += libgpi.pc.in

DISTCLEANFILES = libgpc.pc

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libgpi.pc

if WITH_INFINIBAND
install-data-hook:
	echo "Libs.private: @ac_lib_infiniband@ -libverbs -pthread" >> $(libdir)/pkgconfig/libgpi.pc
else
install-data-hook:
	echo "Libs.private: -lrt -pthread" >> $(libdir)/pkgconfig/libgpi.pc
endif

# CLEAN
CLEANFILES = *~ *.bak *.mod
