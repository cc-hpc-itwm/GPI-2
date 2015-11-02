BIN = loop_barrier.bin loop_barrier_group.bin loop_barrier_group_timeout.bin allreduce.bin \
	barrier_timeout.bin allreduce_user_fun.bin allreduce_utils.bin allreduce_user_type.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
