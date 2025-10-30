******************************************************************************
                       GPI-2
              http://www.gpi-site.com

                  Version: 1.6.0
              Copyright (C) 2013-2025
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
- libfabric (recommended v2.2.0) if support for libfabric is necessary.


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

By default, GPI-2 will be compiled without Infiniband Extensions
support, however the user can also enable and using it (if the header
file is found) by `--enable-infiniband-ext`. Note, however, they are
for the moment an experimental feature.

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
./configure --with-ofi=<full_path_to_ofi_installation>

```


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
  -b <binary file> Use a different binary for first node (master).
                   The master (first entry in the machine file) is
           started with a different application than the rest
           of the nodes (workers).

  -N               Enable NUMA for processes on same node. With this
           option it is only possible to start the same number
           of processes as NUMA nodes present on the system.
           The processes running on same node will be set with
           affinity to the proper NUMA node.

  -n <procs>       Start as many <procs> from machine file.
               This option is used to start less processes than
           those listed in the machine file.

  -d               Run with GDB (debugger) on master node. With this
           option, GDB is started in the master node, to allow
           debugging the application.

  -p               Ping hosts before starting the binary to make sure
           they are available.

  -h               Show help.
```

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


## 6. TROUBLESHOOTING AND KNOWN ISSUES

If there are troubles when building GPI-2 with support for Infiniband,
make sure the OFED stack is correctly installed and running. As above
mentioned, it is possible to specify the OFED path in the actual host
system.

When installing GPI-2 with MPI mixed-mode support (using the options
`--with-mpi` or `--with-mpi<=path_to_mpi_installation>`) and the
installation is failing when trying to build the tests due to missing
libraries, try to setup directly the MPI compilers (wrappers) through
the environment variables CC and FC.


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


## 7. UP COMING FEATURES

GPI-2 is on-going work and more features are still to come. Here are
some that are in our roadmap:

- support to add spare nodes (fault tolerance)
- better debugging possibilities


## 7. LICENSE
GPI-2 is released under the GPL-3 license (see [COPYING](COPYING)).

If you would like to contribute to GPI-2, please get in touch with the
development team at the CC-HPC of the Fraunhofer ITWM, lead by Rui
Machado (contact can be found in the commits log).

## 8. MORE INFORMATION

For more information, check the GPI-2 website ( www.gpi-site.com ).
