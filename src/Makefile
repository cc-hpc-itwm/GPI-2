OFED_PATH = /usr/local/ofed1.51

all:
	gcc -O2 -DNDEBUG -Wall -I$(OFED_PATH)/include -D_GNU_SOURCE -c GPI2.c
	rm -f libGPI2.a
	ar crs libGPI2.a GPI2.o
	cp libGPI2.a ../lib64
	cp GASPI.h ../include

debug:
	gcc -g -O0 -Wall -I$(OFED_PATH)/include -D_GNU_SOURCE -c GPI2.c
	rm -f libGPI2-dbg.a
	ar crs libGPI2-dbg.a GPI2.o
	cp libGPI2-dbg.a ../lib64
	cp GASPI.h ../include

mic:
	icc -mmic -O2 -DNDEBUG -Wall -I$(OFED_PATH)/include -DMIC -D_GNU_SOURCE -c GPI2.c
	rm -f libGPI2.a
	ar crs libGPI2.a GPI2.o
	cp libGPI2.a ../lib64/mic
	cp GASPI.h ../include/mic

clean:
	rm -f *.o libGPI2.a libGPI2-dbg.a *~ ../lib64/libGPI2* ../lib64/mic/libGPI2.a
