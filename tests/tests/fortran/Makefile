BIN =  fortran_use.bin fortran_allreduces.bin fortran_seg_ptr.bin fortran_seg_use.bin

FCFLAGS+=-I$(GPI_DIR)/include -fno-range-check

build: $(BIN)

%.bin: %.f90
	@if test "$(FC)" != ""; then \
	$(FC) $(FCFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS); \
	fi

clean:
	rm -rf *~ \#_* *.o *.bin *.mod
