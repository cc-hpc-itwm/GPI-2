include ../make.defines

LIBS_BENCH = $(subst -lGPI2-dbg,-lGPI2, $(LIBS))
BIN = write_bw.bin write_lat.bin read_bw.bin ping_pong.bin barrier.bin nb_barrier.bin \
	allreduce.bin nb_allreduce.bin

build: $(BIN)

%.bin:  %.o common.o
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS_BENCH)
clean:
	rm -rf *~ \#_* *.o *.bin
