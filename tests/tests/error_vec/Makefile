BIN =  get_error_vec.bin get_error_vec_fail.bin one_dies.bin \
	barrier_recover.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

#barrier_recover.bin: barrier_recover.c
#	$(CXX) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)
%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
