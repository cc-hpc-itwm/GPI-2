# INCLUDES
include $(top_srcdir)/amdoxygen.am
ACLOCAL_AMFLAGS = -I m4

# TARGETS
SUBDIRS = src tests docs

# FOR DISTRIBUTION
dist_bin_SCRIPTS = bin/*
EXTRA_DIST = autogen.sh tutorial $(DX_CONFIG)

# TESTING
check :
	cd tests/; ./runtests.sh

# CLEAN
MOSTLYCLEANFILES = $(DX_CLEANFILES)
CLEANFILES = *~ *.bak m4/*~
