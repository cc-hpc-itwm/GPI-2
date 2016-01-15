include make.inc

ifndef GPI2_DEVICE
GPI2_DEVICE = IB
$(info Default configuration (Infiniband))
endif

ifeq ($(findstring IB,$(GPI2_DEVICE)),IB)
$(info Configuration for Infiniband)
include devices/ib/Makefile.inc
CFLAGS+=-DGPI2_DEVICE_IB
DBG_CFLAGS+=-DGPI2_DEVICE_IB
MIC_CFLAGS+=-DGPI2_DEVICE_IB
else ifeq ($(findstring TCP,$(GPI2_DEVICE)),TCP)
$(info Configuration for Ethernet)
include devices/tcp/Makefile.inc
CFLAGS+=-DGPI2_DEVICE_TCP
DBG_CFLAGS+=-DGPI2_DEVICE_TCP
else
$(error Unknow interconnect $(GPI2_DEVICE))
endif

SRCS += GPI2_Env.c
SRCS += GPI2_Utility.c
SRCS += GPI2_SN.c
SRCS += GPI2_Logger.c
SRCS += GPI2_Stats.c
SRCS += GPI2_Mem.c
SRCS += GPI2_Threads.c
SRCS += GPI2_Coll.c
SRCS += GPI2_IO.c
SRCS += GPI2_ATOMIC.c
SRCS += GPI2_PASSIVE.c
SRCS += GPI2_SEG.c
SRCS += GPI2_GRP.c
SRCS += GPI2_CONFIG.c
SRCS += GPI2_CM.c
SRCS += GPI2.c

HDRS += GPI2_Env.h
HDRS += GPI2_Utility.h
HDRS += GPI_Types.h
HDRS += GPI2_SN.h
HDRS += GPI2_Coll.h
HDRS += GPI2_CM.h
HDRS += GPI2.h

OBJS = $(SRCS:.c=.o)
OBJS_DBG = $(SRCS:.c=.dbg.o)
OBJS_MIC = $(SRCS:.c=.mic.o)
OBJS_DBG_MIC = $(SRCS:.c=.mic.dbg.o)

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
	$(FORTRANC) -O2 -I. -c $<  GASPI_Ext.f90 
	cp gaspi.mod gaspi_types.mod gaspi_ext.mod ../include;

mic: $(OBJS_MIC) $(OBJS_DBG_MIC)
	rm -f ../lib64/mic/$(GPI2) ../lib64/mic/$(GPI2_DBG)
	ar crs ../lib64/mic/$(GPI2) $(OBJS_MIC)
	ar crs ../lib64/mic/$(GPI2_DBG) $(OBJS_DBG_MIC)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

%.dbg.o: %.c
	$(CC) $(DBG_CFLAGS) -DDEBUG $(INCLUDES) -c $<  -o $@

%.mic.o: %.c
	icc -mmic $(MIC_CFLAGS) $(INCLUDES) -DMIC -c $<  -o $@

%.mic.dbg.o: %.c
	icc -mmic -g -O0 $(INCLUDES) -DGPI2_DEVICE_IB -D_GNU_SOURCE -DDEBUG -DMIC -c $<  -o $@

clean:
	$(RM) -r $(OBJ_FORTRAN) $(OBJS) $(OBJS_DBG) *~
	$(RM) -r $(OBJS_MIC) $(OBJS_DBG_MIC)
	$(RM) ../lib64/$(GPI2) ../lib64/$(GPI2_DBG) ../lib64/mic/$(GPI2)
	$(RM) *.mod ../include/gaspi_types.mod ../include/gaspi.mod

depend: $(SRCS)
	makedepend $(INCLUDES) $^

.PHONY: depend clean

# DO NOT DELETE THIS LINE -- make depend needs it
