BIN = with_mpi.bin

CFLAGS+=-I../ -I$(HOME)/soft_install/openmpi/icc14/1.6.5/include -L$(HOME)/soft_install/openmpi/icc14/1.6.5/lib
CC=$(HOME)/soft_install/openmpi/icc14/1.6.5/bin/mpicc

build: $(BIN)

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $(TOPROOT)/bin/$@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
