SUBDIRS = src tests

ACLOCAL_AMFLAGS = -I m4

dist_bin_SCRIPTS = bin/*

EXTRA_DIST = autogen.sh tests/tests/compile_rules.in tests/microbenchmarks/compile_rules.in


# TESTS
# include tests/Makefile.am.inc

# check_PROGRAMS = $(test_BUILT_TESTS)
# TESTS = $(test_BUILT_TESTS)
# LOG_DRIVER = cd tests; ./runtests.sh -m machinefile

check :
	$(shell cd tests/; ./runtests.sh)

CLEANFILES = *~ *.bak

# all: gpi
# #fortran tests

# gpi:
# $(MAKE) -C src gpi

# fortran:
# $(MAKE) -C src fortran

# mic:
# $(MAKE) -C src mic

# tests: gpi
# cd tests && $(MAKE) && cd ..

# docs:
# @if test "$(DOXYGEN)" = ""; then \
# echo "Doxygen not found."; \
# echo "Install doxygen to be able to generate documentation."; \
# echo "Or consult it online at: http://www.gpi-site.com/gpi2/docs/";\
# false; \
# fi
# doxygen Doxyfile

# clean:
# $(MAKE) -C src clean
# $(MAKE) -C tests clean

# .PHONY: all tests docs clean
