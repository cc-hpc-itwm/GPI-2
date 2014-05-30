include ../make.defines

BIN = write_bw.bin write_lat.bin read_bw.bin ping_pong.bin barrier.bin nb_barrier.bin \
	allreduce.bin

build: $(BIN)

%.bin:  %.o common.o
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)
clean:
	rm -rf *~ \#_* *.o *.bin
