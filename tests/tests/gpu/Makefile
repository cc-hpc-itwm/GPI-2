BIN =  hello_gpus.bin seg_alloc_gpu.bin seg_gpu_register.bin write_gpu.bin write_gpu_write.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin

.PHONY: clean