SUBDIRS = fortran init config groups collectives comm passive atomic lists segments notifications error_vec \
	threads utils #mpi

.PHONY: subdirs $(SUBDIRS) clean

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	$(foreach dir, $(SUBDIRS), $(MAKE) -C $(dir) clean;)
	rm -f *.o *~
