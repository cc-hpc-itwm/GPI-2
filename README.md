******************************************************************************
                       GPI-2
              http://www.gpi-site.com

                  Version: 1.7.0
              Copyright (C) 2013-2026
                 Fraunhofer ITWM

******************************************************************************

## 1. INTRODUCTION

GPI-2 is the second generation of GPI (www.gpi-site.com). GPI-2
implements the GASPI specification (www.gaspi.de), an API
specification which originates from the ideas and concepts of GPI.

GPI-2 is an API for asynchronous communication. It provides a
flexible, scalable and fault tolerant interface for parallel
applications.


## 2. INSTALLATION


### Requirements

The current version of GPI-2 has the following requirements.

Software:
- libibverbs v1.1.6 (Verbs library from OFED) if running on Infiniband.
- ssh server running on compute nodes, requiring no password, if
  running with ssh support (default).
- autotools utilities (autoconf>=2.63,libtool>=2.2,automake>=1.11)
- libfabric if support for libfabric is necessary; v2.7.0 or later is
  recommended: earlier versions have bugs in the `ofi_rxm` provider
  (crash at termination) and the `shm` provider (occasional hang at
  start) that GPI-2 applications can hit.


### Basic configuration

If GPI-2 is cloned from the repository, it is necessary to generate
the files and scripts required for its configuration. This is achieved
by the command line:

```
./autogen.sh
```

After this step, the configuration is done using the script
`./configure`.  The available options and the relevant environment
variables are printed by `./configure --help`.  The basic
configuration:

```
./configure --prefix=$HOME/local
```

uses the compilers defined by the environment variables CC and FC for
the general checking procedure and sets up `$HOME/local` as the
installation directory. By default, the script:

- checks for the Infiniband header and library files, and fall backs
to the Ethernet device in case they are not available or usable,
- targets to the production, debugging and statistic libraries (both
static and shared), as well as, the Fortran modules (if the Fortran
compilers are found),
- configures GPI-2 to use ssh for application start,
- checks the existence of `doxygen` and `dot` for the documentation target.

In the case autotools is not available in the target system, the user
can create a tarball distribution of GPI-2 in a system with autotools
by:

`./autogen.sh ; ./configure ; make dist`

After unpacking the distribution `gpi-2*.tar.gz` in the target system,
GPI-2 can be configured and compiled.

### Compilation, testing and cleaning:

The compilation step:

```
make -j$NPROC
```

builds in parallel the GPI-2 libraries, the Fortran modules, and the
binary tests and microbenchmarks.  After successful completion, the
user can define the working hosts in `tests/machines` and run the
predefined tests by:

```
make check
```

or by using an environment variable for the working hosts, e.g.:

```
GPI2_RUNTEST_OPTIONS="-m ~/my_machines" make check
```

(see more options in [tests usage](tests/README)).

Cleaning of the configuration/compilation files can be done as usual
with the commands `make distclean` and `make clean`.

### Documentation and tutorial

If required, the (doxygen) documentation and the tutorial code are
built through `make docs` and `make tutorial`, respectively.

### Installation and uninstallation

Finally, `make install` installs:

- the running scripts in the `$PREFIX/bin` directory,
- the shared and static libraries in `$PREFIX/lib64`,
- the headers and Fortran modules in `$PREFIX/include`,
- the full tests directory in `$PREFIX/tests`

Where `$PREFIX` refers to the path provided with the `--prefix` option
in `configure`. If not path is provided, and by default, the location
is `/usr/local/`

Note, as usual, the path to the GPI-2 shared libraries need to be
added to the `LD_LIBRARY_PATH` environment variable:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PREFIX/lib64
```

If required the package can be removed from the target directory by
using `make uninstall`.

### Custom configurations

Specific configurations can be setup by predefined flags.

#### DEVICES

##### Infiniband
GPI-2 is intended to be linked to the libibverbs from the OFED stack.
In case the configure script is not able to find it in the default paths
of the host system, the user can pass the path of the OFED
installation:

```
./configure --with-infiniband<=full_path_to_ofed>
```

###### Dynamically Connected transport

`GASPI_IB_DCT=1` selects the Dynamically Connected (DC) transport of
NVIDIA (Mellanox) NICs instead of RC: a fixed number of queue pairs per
process instead of a number growing with the job size. It is built if
`mlx5dv` (rdma-core) is found (`./configure gpi2_cv_ib_dc=no` to disable);
if DC is not available, `gaspi_proc_init` fails. All processes must use the
same setting (`gaspi_run` forwards it):

```
GASPI_IB_DCT=1 gaspi_run -m machinefile my_app
```

Each queue and the collectives use up to 16 DC initiators, created when
first used (about 260 KiB each); reads from 16 KiB go in up to four parts.

With DC, a request to an unreachable process (e.g. a dead one) also fails
the requests behind it on the same initiator, i.e. to processes sharing it;
`gaspi_state_vec_get` marks only the unreachable process. Applications
that must tolerate failed processes should keep RC.

The former option `--enable-infiniband-ext` was removed.

With several InfiniBand NICs and no `netdev_id` in the configuration, a
process uses a NIC on the NUMA node of its CPUs. `GASPI_IB_VERBOSE=1` prints
each process's device and the NUMA nodes of the device and the CPUs.

##### Ethernet / TCP
On the other hand, GPI-2 can be installed on a system without
Infiniband, using standard TCP sockets:

```
./configure --with-ethernet
```

Such support is, however, primarily targetted at the development of
GPI-2 applications without the need to access a system with
Infiniband, **with less focus on performance**.

##### Other interconnects and fabrics (libfabric)

Support for other devices and fabrics such as Amazon EFA, Cray
Slingshot or Omnipath is provided by libfabric, the Open Fabrics
Interfaces (ofi), framework.

GPI-2 aims to support all of the different fabric providers available
with libfabric. Currently this is **not yet** completely tested on all
the different providers and some issues may exist.

To configure GPI-2 with libfabric
```
./configure --with-ofi
```

If a libfabric version is found, that is enough. If however, it cannot
be found, one can provide a path to the option:

```
./configure --with-ofi=<full_path_to_libfabric_installation>

```

libfabric chooses the provider between nodes; `FI_PROVIDER` sets it (e.g.
`cxi`, `verbs;ofi_rxm`). Processes on the same node use `shm`. With several
cxi or InfiniBand NICs, a process bound to one NUMA node uses the NIC of
that node. Environment variables of the device:

| Variable | Meaning |
|----------|---------|
| `GASPI_OFI_DEVICE` | Use this device (e.g. `cxi2`). |
| `GASPI_OFI_VERBOSE` | If set, print each process's provider and device at start. |
| `GASPI_OFI_PROGRESS_PAUSE_US` | Pause between passes of a progress thread, in microseconds (default 10). |
| `GASPI_OFI_PROGRESS_WAIT_PAUSE_US` | The same while the application waits for completions and drives progress itself (default 50). |
| `GASPI_OFI_CONNECT_TIMEOUT_MS` | Wait for a process that does not respond before checking whether it is alive, in milliseconds (default 1000). |
| `GASPI_OFI_UNSIGNALED_WRITES` | If `0`, each small write completes on its own instead of in `gaspi_wait`: a lower rate, but with cxi errors name the target rank. |
| `GASPI_OFI_SHM` | If `0`, processes on the same node do not use `shm` but the provider between nodes. |
| `GASPI_OFI_SHM_PTRACER` | If `0`, the user's other processes cannot ptrace the process, which shm's direct copies (CMA) need under Yama `ptrace_scope` 1: shm latency is then 3 to 12 times higher. |
| `GASPI_OFI_SHM_FAST_RMA` | If `1`, shm does writes and reads as system calls in the caller instead of commands for the target: faster around 4 KB, much slower for small messages. |

`gaspi_run` passes these variables to all processes. Failed processes are
reported as with Infiniband. On HPE Cray systems with Slingshot (`cxi`),
configure GPI-2 with PALS (see below).


#### BATCH SYSTEM
By default, GPI-2 uses `ssh` to initialize the application on the
chosen/provided nodes. However, the user can configure it to use
Slurm:

```
./configure --with-slurm
```

or LoadLeveler:

```
./configure --with-loadleveler
```

or PALS, the launcher of HPE Cray systems with PBS Pro:

```
./configure --with-pals
```

#### MPI Interoperability

If the plan is to use GPI-2 with MPI to, for instance, start an
incremental port of a large application or to use some libraries that
require MPI, the user can enable MPI interoperability in several ways:

- checking for MPI in the standard path: `./configure --with-mpi`
- checking for MPI in a specific path, e.g.: `./configure --with-mpi=<=path_to_mpi_installation>`
- specifying the MPI compilers, e.g.: `CC=mpicc FC=mpif90 ./configure`

For this MPI+GPI-2 mixed mode, the only constraint is that **MPI_Init()
must be invoked before gaspi_proc_init()** and it is assumed that the
application starts with mpirun (or mpiexec, etc.). Also, note that
this option will require that the GPI-2 application is linked to the MPI
library (even if MPI is not used). Therefore, if the interest is to
use GPI-2 only, GPI-2 must not be build with this option.

Furthermore, fine control of MPI can be done through the
`--with-mpi-extra-flags` option. For example, to configure with Intel
MPI compilers and link to the thread safe version of the Intel MPI
Library:

```
CC=mpiicc FC=mpiifort ./configure --with-mpi-extra-flags=-mt_mpi
```

#### GPU/CUDA interoperability

GPI-2 allows a direct data transfer between NVIDIA GPUs through Mellanox
HCA and the GPUDirectRDMA's API. To this end the system must satisfy
the following requirements:

- InfiniBand or RoCE adapter cards with Mellanox ConnectX-4 (or later)
  technology,
- Kepler, Tesla or Quadro GPUs
- NVIDIA software components (CUDA 5.0 or above),
- A properly loaded GPUDirect kernel module on each of the compute
nodes (can be verified through `service nv_peer_mem status` or `lsmod
| grep nv_peer_mem `)

There is neither special configuration and/or compilation setup for
GPI-2 nor special GASPI/GPI-2 functions to use GPUs and/or
GPUdirectRDMA. The user just needs to properly allocate the memory
segments and buffers into the host(s)/device(s) using the GPI-2 and
CUDA APIs. Specific considerations about the memory management and
general design of applications using GPUdirectRDMA can be found in
[https://docs.nvidia.com/cuda/gpudirect-rdma/index.html].

## 3. BUILDING GPI-2 APPLICATIONS

By default, GPI-2 provides two libraries: `libGPI2.a` and `libGPI2-dbg.a`,
and their corresponding shared versions: `libGPI2.so` and `libGPI2-dbg.so`.

The `libGPI2.*` aims at high-performance and is to be used in production
whereas the `libGPI2-dbg.*` provides a debug version, with extra
parameter checking and debug messages and is to be used to debug and
during development.

There is also `libGPI2-stats.*` which prints some statistics about
operations at `gaspi_proc_term`. It is useful to get an impression of
which and how often operations where invoked to pinpoint some
performance bottlenecks.

## 4. RUNNING GPI-2 APPLICATIONS

The `gaspi_run` utility is used to start and run GPI-2
applications. A machine file with the hostnames of nodes where the
application will run, must be provided.

For example, to start 1 process per node (on 4 nodes), the machine
file looks like:

```
node01
node02
node03
node04
```

Similarly, to start 2 processes per node (on 4 nodes):

```
node01
node01
node02
node02
node03
node03
node04
node04
```

The `gaspi_run` utility is invoked as follows:

```
gaspi_run -m <machinefile> [OPTIONS] <path GASPI program>
```

**IMPORTANT: The path to the program must exist on all nodes where the
program should be started.**

There are however a couple of exceptions that do not require the usage
of a machine file (-m option):
1. if GPI-2 was configured to use Slurm as the batch system. In this
  case, the allocated nodes from Slurm will be used and there is no
  need to pass the machine file.
2. to run processes on a local host. In this case, it is only
   necessary to provide the number of processes using `-n N`
   option. This will run the number of process on the current
   hostname.

The `gaspi_run` utility has the following further options `[OPTIONS]`:

```
  -b, --binary <binary file>
                   Use a different binary for first node (master).
                   The master (first entry in the machine file) is
                   started with a different application than the rest
                   of the nodes (workers).

  -N, --NUMA       Enable NUMA for processes on same node. The
                   processes running on same node are bound to the
                   NUMA nodes with CPUs in turn: local process k to the
                   (k mod n)-th of the n NUMA nodes, starting with
                   the first NUMA node with a NIC. With more processes
                   than NUMA nodes, processes share a NUMA node. A
                   process bound already (launcher, numactl, taskset,
                   cpuset) keeps its binding, with a warning. Needs
                   numactl or lscpu on the nodes.

  -n, --nodes <procs>
                   Start as many <procs> from machine file.
                   This option is used to start less processes than
                   those listed in the machine file.

  -d, --debug      Run with GDB (debugger) on master node. With this
                   option, GDB is started in the master node, to allow
                   debugging the application.

  -p, --ping       Ping hosts before starting the binary to make sure
                   they are available.

  -h, --help       Show help.
```

### Binding to a CPU near the NIC

`GASPI_BIND=nic-core` binds each process to one CPU on a NUMA node with a
NIC: local process k to the k-th CPU, taking one hardware thread per core
first and the NUMA nodes with a NIC in turn. This gives the lowest latency
and highest message rate for communication-bound runs; applications that
need memory bandwidth or many cores per process are usually better served
by `-N`. `GASPI_BIND=nic-core` takes precedence over `-N` and, like `-N`,
leaves a process bound already as it is. `GASPI_BIND=none` (the default)
binds nothing.

The microbenchmarks in `tests/microbenchmarks` use `GASPI_BIND=nic-core`
unless `GASPI_BIND` is set, and print the setting.

### Running with PALS (HPE Cray, PBS Pro)

In a PBS job, `gaspi_run` uses the job's nodes if no machine file is given.
The entries of a node must be consecutive, and all nodes but the last need
the same number of processes. By default, the processes of a node are
spread over its NUMA nodes and use the NIC of their NUMA node; `-N` and
`GASPI_BIND` therefore keep that binding.
`GPI2_PALS_ARGS` passes further `mpiexec` options; a CPU binding there
replaces the default, e.g. for hybrid (OpenMP) programs:

```
GPI2_PALS_ARGS="--depth 24 --cpu-bind depth" gaspi_run -m machinefile my_app
```

Keep each process within one NUMA node: unbound processes communicate
much more slowly.

`gaspi_cleanup -m machinefile` kills the last program started on the hosts
of the machine file; with PALS it uses `mpiexec` and runs within the job.
In the installed tests directory, `runtests.sh` uses this `gaspi_run` and
`gaspi_cleanup`: a machine file in `tests/machines` is all it needs.

### Non-interactive usage

`gaspi_run` can of course be used used in a batch job. In general, the
information required to setup such file job scheduler can be obtained
from environment variables defined by the job scheduler. The directory
`docs/batch_examples` includes sample scripts for setting the machine
file and submitting jobs to common batch processing systems. They can
be used as starting point for some elaborated applications and
particular environments.

## 5. THE GASPI_LOGGER

The `gaspi_logger` utility is used to view and separate the output
from all nodes when the function gaspi_printf is called. The
`gaspi_logger` is started, on another session or in the backgroun, on
the master node. The output of the application, when using
gaspi_printf, will be redirected to the `gaspi_logger`. Other I/O
routines (e.g. printf) will not.

A further separation of output (useful for debugging) can be achieved
by using the routine `gaspi_printf_to` which sends the output to the
`gaspi_logger` started on a particular node. For example,

```
gaspi_printf_to(1, "Hello 1\n");
```

will display the string "Hello 1" in the `gaspi_logger` started on rank
1.


## 6. TRACING

GPI-2 includes a built-in tracing facility for profiling GASPI
operations. Tracing is activated at runtime via the `GASPI_TRACE`
environment variable — no recompilation needed for lifecycle
operations (init, segments, groups, collectives, passive).

Data-path operations (read, write, notify, wait, atomics) are only
traced when linked against `libGPI2-stats`, which includes full
hot-path instrumentation.

### Enabling tracing

Using `gaspi_run`:

```
gaspi_run -m machinefile --trace my_app           # trace all categories
gaspi_run -m machinefile --trace init,seg my_app  # trace only init and segments
```

Or directly via environment variable:

```
GASPI_TRACE=all gaspi_run -m machinefile my_app
GASPI_TRACE=init,coll gaspi_run -m machinefile my_app
```

Available categories: `init`, `seg`, `io`, `coll`, `grp`, `atomic`,
`passive`, `sn`. Use `all` or `1` to enable everything.

### Trace output

Each rank writes a binary trace file `gaspi_trace_rank<N>.bin` in
the working directory. Use the `gpi2_trace_dump` tool to convert:

```
gpi2_trace_dump gaspi_trace_rank0.bin                  # human-readable text
gpi2_trace_dump -f csv gaspi_trace_rank0.bin           # CSV
gpi2_trace_dump -f json gaspi_trace_rank0.bin > t.json # Chrome Trace Format
```

Multiple files can be merged into a single output — events are
sorted by timestamp and tagged with their rank:

```
gpi2_trace_dump gaspi_trace_rank*.bin                   # merged text
gpi2_trace_dump -f json gaspi_trace_rank*.bin > all.json # merged JSON
gpi2_trace_dump -r gaspi_trace_rank*.bin                # relative (each rank starts at t=0)
```

The JSON output can be visualized in `chrome://tracing` or
[Perfetto UI](https://ui.perfetto.dev).


## 7. TROUBLESHOOTING AND KNOWN ISSUES

If there are troubles when building GPI-2 with support for Infiniband,
make sure the OFED stack is correctly installed and running. As above
mentioned, it is possible to specify the OFED path in the actual host
system.

When installing GPI-2 with MPI mixed-mode support (using the options
`--with-mpi` or `--with-mpi<=path_to_mpi_installation>`) and the
installation is failing when trying to build the tests due to missing
libraries, try to setup directly the MPI compilers (wrappers) through
the environment variables CC and FC.


### Infiniband with relaxed PCIe ordering

On NVIDIA (Mellanox) NICs the firmware setting `PCI_WR_ORDERING=1`
(`force_relax`) makes all writes of the NIC to host memory relaxed-ordered. With
it, a notification may become visible before the data transferred just before
it, so the application can read stale data. This affects `gaspi_notify` after
`gaspi_write`, `gaspi_write_notify` and `gaspi_write_list_notify` at the target,
and `gaspi_read_notify` and `gaspi_read_list_notify` at the initiator. It also
affects `gaspi_allreduce`, which may compute its result from stale data. The
default setting (`PCI_WR_ORDERING=0`, `per_mkey`) is not affected.

If any node of a job uses `force_relax`, set `GASPI_IB_RELAXED_ORDERING=1` when
starting the application:

```
GASPI_IB_RELAXED_ORDERING=1 gaspi_run -m machinefile my_app
```

With the libfabric device, the variable applies to `verbs;ofi_rxm` and
requires `group_max <= 128`.

Applications that signal the arrival of data with their own flag, written with
`gaspi_write` after the data, instead of a notification remain affected, also
with the variable set: GASPI only guarantees that the data written before a
notification on the same queue has arrived when the notification is seen.

GPI-2 cannot detect the NIC setting itself, so system administrators of affected
clusters should set the variable for all users, for example in the GPI-2
environment module.


### Environment variables

You might have some trouble when your application requires some
dynamically set environment setting (e.g. the `LD_LIBRARY_PATH`), for
instance, through the module system of your jobs batch
system. Currently, neither the `gaspi_run` or the GPI-2 library take
care of such environment settings. To this situation there are 2
workarounds:

i) you set the required environment variables in your shell
initialization file (e.g. ~/.bashrc).

ii) you create an executable shell script which sets the required
environment variables and then starts the application. Then you can
use `gaspi_run` to start the application, providing the shell script as
the application to execute.

```
gaspi_run -m machinefile ./my_wrapper_script.sh
```

where `my_wrapper_script.sh` contains:

```shell
#!/bin/sh

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_my_lib>

<path_to_my_application>/my_application <my_app_args>

exit $?
```

If you're running in MPI mixed-mode, starting your application with
mpirun/mpiexec, this should not be an issue.
