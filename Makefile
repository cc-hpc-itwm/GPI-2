DOXYGEN:=$(shell which doxygen)
GFORTRAN:=$(shell which gfortran)

all: gpi tests docs 

gpi:
	make -C src
	make -C src debug
	@if test "$(GFORTRAN)" != ""; then \
	make -C src fortran; \
	fi	

mic:
	make -C src mic

tests: 
	cd tests; make; cd ..

docs:
	@if test "$(DOXYGEN)" = ""; then \
		echo "Doxygen not found."; \
		echo "Install doxygen to be able to generate documentation."; \
		echo "Or consult it online at: http://www.gpi-site.com/gpi2/docs/";\
		false; \
	fi
	doxygen Doxyfile

clean:
	make -C src clean
	make -C tests clean

.PHONY: all tests docs clean 
