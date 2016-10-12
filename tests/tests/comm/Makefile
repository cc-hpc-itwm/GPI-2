BIN = write_simple.bin write_all_nsizes.bin write_all_nsizes_mtt.bin	\
	write_all_nsizes_nobuild.bin write_timeout.bin			\
	read_nsizes.bin comm_limits.bin all-to-all.bin			\
	all-to-rank0.bin z4k_pressure.bin z4k_pressure_mtt.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
