SUBDIRS = fortran init config groups collectives comm passive atomic lists segments notifications error_vec \
	threads utils #mpi

.PHONY: subdirs $(SUBDIRS) clean

install: subdirs
	$(foreach dir, $(SUBDIRS), cp $(dir)/*.bin ../bin;)

subdirs: $(SUBDIRS)

$(SUBDIRS): test_utils.o
	$(MAKE) -C $@


%.o: %.c
	$(CC) $(CFLAGS) -I. -c $^

clean:
	$(foreach dir, $(SUBDIRS), $(MAKE) -C $(dir) clean;)
	rm -f *.o *~
