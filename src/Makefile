OFED_PATH = /usr/local/ofed1.51

all:
	gcc -O2 -Wall -I$(OFED_PATH)/include -D_GNU_SOURCE -c GPI2.c
	rm -f libGPI2.a
	ar crs libGPI2.a GPI2.o
	cp libGPI2.a ../lib64
	cp GASPI.h ../include
	cp PGASPI.h ../include

debug:
	gcc -DDEBUG -g -O0 -Wall -I$(OFED_PATH)/include -D_GNU_SOURCE -c GPI2.c
	rm -f libGPI2-dbg.a
	ar crs libGPI2-dbg.a GPI2.o
	cp libGPI2-dbg.a ../lib64
	cp GASPI.h ../include
	cp PGASPI.h ../include

fortran:
	gfortran -O2 -c GASPI.f90
	cp gaspi.mod gaspi_types.mod ../include

mic:
	icc -mmic -O2 -Wall -I$(OFED_PATH)/include -DMIC -D_GNU_SOURCE -c GPI2.c
	rm -f libGPI2.a
	ar crs libGPI2.a GPI2.o
	cp libGPI2.a ../lib64/mic
	cp GASPI.h ../include/mic
	cp PGASPI.h ../include/mic

clean:
	rm -f *.o libGPI2.a libGPI2-dbg.a *~ ../lib64/libGPI2* ../lib64/mic/libGPI2.a ../include/GASPI.h ../include/PGASPI.h
	rm -f *.mod ../include/gaspi_types.mod ../include/gaspi.mod
