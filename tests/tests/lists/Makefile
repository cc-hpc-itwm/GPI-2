BIN = write_list.bin write_list_all.bin write_list_check.bin read_list_check.bin #read_list.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
