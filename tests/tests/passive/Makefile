BIN = passive.bin passive_n.bin passive_nbytes.bin			\
	passive_recvtimeout.bin passive_own.bin passive_own_block.bin	\
	passive_tout_test.bin passive_limits.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
