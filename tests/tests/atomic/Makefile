BIN = fetch_add.bin compare_swap.bin fetch_add_test.bin compare_swap_test.bin  \
	fetch_add_loop.bin fetch_add_unalignd.bin fetch_add_nobuild.bin \
	compare_swap_nobuild.bin atomic_fail_no_seg.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
