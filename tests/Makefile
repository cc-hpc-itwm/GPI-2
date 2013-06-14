include make.defines

SUBDIRS = tests microbenchmarks

.PHONY: subdirs $(SUBDIRS) clean

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	$(MAKE) -C tests clean
	$(MAKE) -C microbenchmarks clean
	rm -f *~ *.o
	rm -f bin/*.bin
