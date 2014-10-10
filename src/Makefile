include make.inc

SRCS += GPI2_IB_IO.c GPI2_IB_PASSIVE.c GPI2_IB_ATOMIC.c GPI2_IB_GRP.c GPI2_IB_SEG.c GPI2_IB_CONFIG.c GPI2_IB.c \
 GPI2_Env.c GPI2_Utility.c GPI2_SN.c GPI2_Logger.c GPI2_Stats.c GPI2_Mem.c GPI2_Threads.c GPI2.c 
HDRS += GPI2_IB.h GPI2_Env.h GPI2_Utility.h GPI_Types.h GPI2_SN.h GPI2.h

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
	$(FORTRANC) -O2 -c $< 
	cp gaspi.mod gaspi_types.mod ../include;

mic: $(OBJS_MIC) $(OBJS_DBG_MIC)
	rm -f ../lib64/mic/$(GPI2) ../lib64/mic/$(GPI2_DBG)
	ar crs ../lib64/mic/$(GPI2) $(OBJS_MIC)
	ar crs ../lib64/mic/$(GPI2_DBG) $(OBJS_DBG_MIC)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

%.dbg.o: %.c
	$(CC) $(DBG_CFLAGS) -DDEBUG $(INCLUDES) -c $<  -o $@

%.mic.o: %.c
	icc -mmic $(MIC_CFLAGS) -DMIC -c $<  -o $@

%.mic.dbg.o: %.c
	icc -mmic -g -O0 $(INCLUDES) -D_GNU_SOURCE -DDEBUG -DMIC -c $<  -o $@

clean:
	$(RM) *.o *~ ../lib64/$(GPI2) ../lib64/$(GPI2_DBG) ../lib64/mic/$(GPI2)
	$(RM) *.mod ../include/gaspi_types.mod ../include/gaspi.mod

depend: $(SRCS)
	makedepend $(INCLUDES) $^

.PHONY: depend clean

# DO NOT DELETE THIS LINE -- make depend needs it
