BIN = write.bin write_simple.bin write_all.bin write_all_mtt.bin write_all_nsizes.bin \
	write_all_nsizes_mtt.bin write_timeout.bin big_transfers.bin \
	z4k_pressure.bin z4k_pressure_mtt.bin read_all_nsizes.bin read_smalls.bin \
	strings.bin

CFLAGS+=-I../

build: $(BIN)

%.bin: ../test_utils.o  %.o 
	$(CC) $(CFLAGS) $(LIB_PATH) -o $(TOPROOT)/bin/$@ $^ $(LIBS)
clean:
	rm -rf *~ \#_* *.o *.bin
