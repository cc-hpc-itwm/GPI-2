BIN = with_mpi.bin mpi_allreduce.bin mpi_barrier.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS) -lm

clean:
	rm -rf *~ \#_* *.o *.bin
