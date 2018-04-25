DOXYGEN:=$(shell which doxygen)
GFORTRAN:=$(shell which $FC)

all: gpi fortran tests

gpi:
	$(MAKE) -C src gpi

fortran:
	$(MAKE) -C src fortran

mic:
	$(MAKE) -C src mic

tests: gpi
	cd tests && $(MAKE) && cd ..

docs:
	@if test "$(DOXYGEN)" = ""; then \
		echo "Doxygen not found."; \
		echo "Install doxygen to be able to generate documentation."; \
		echo "Or consult it online at: http://www.gpi-site.com/gpi2/docs/";\
		false; \
	fi
	doxygen Doxyfile

clean:
	$(MAKE) -C src clean
	$(MAKE) -C tests clean

.PHONY: all tests docs clean 
