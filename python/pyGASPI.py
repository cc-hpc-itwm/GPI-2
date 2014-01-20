__docformat__ =  'restructuredtext'

# Begin preamble

import ctypes, os, sys
from ctypes import *

_int_types = (c_int16, c_int32)
if hasattr(ctypes, 'c_int64'):
    # Some builds of ctypes apparently do not have c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (c_int64,)
for t in _int_types:
    if sizeof(t) == sizeof(c_size_t):
        c_ptrdiff_t = t
del t
del _int_types

class c_void(Structure):
    # c_void_p is a buggy return type, converting to int, so
    # POINTER(None) == c_void_p is actually written as
    # POINTER(c_void), so it can be treated as a real pointer.
    _fields_ = [('dummy', c_int)]

def POINTER(obj):
    p = ctypes.POINTER(obj)

    # Convert None to a real NULL pointer to work around bugs
    # in how ctypes handles None on 64-bit platforms
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

class UserString:
    def __init__(self, seq):
        if isinstance(seq, basestring):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq)
    def __str__(self): return str(self.data)
    def __repr__(self): return repr(self.data)
    def __int__(self): return int(self.data)
    def __long__(self): return long(self.data)
    def __float__(self): return float(self.data)
    def __complex__(self): return complex(self.data)
    def __hash__(self): return hash(self.data)

    def __cmp__(self, string):
        if isinstance(string, UserString):
            return cmp(self.data, string.data)
        else:
            return cmp(self.data, string)
    def __contains__(self, char):
        return char in self.data

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.__class__(self.data[index])
    def __getslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, basestring):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other))
    def __radd__(self, other):
        if isinstance(other, basestring):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other) + self.data)
    def __mul__(self, n):
        return self.__class__(self.data*n)
    __rmul__ = __mul__
    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self): return self.__class__(self.data.capitalize())
    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))
    def count(self, sub, start=0, end=sys.maxint):
        return self.data.count(sub, start, end)
    def decode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())
    def encode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())
    def endswith(self, suffix, start=0, end=sys.maxint):
        return self.data.endswith(suffix, start, end)
    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))
    def find(self, sub, start=0, end=sys.maxint):
        return self.data.find(sub, start, end)
    def index(self, sub, start=0, end=sys.maxint):
        return self.data.index(sub, start, end)
    def isalpha(self): return self.data.isalpha()
    def isalnum(self): return self.data.isalnum()
    def isdecimal(self): return self.data.isdecimal()
    def isdigit(self): return self.data.isdigit()
    def islower(self): return self.data.islower()
    def isnumeric(self): return self.data.isnumeric()
    def isspace(self): return self.data.isspace()
    def istitle(self): return self.data.istitle()
    def isupper(self): return self.data.isupper()
    def join(self, seq): return self.data.join(seq)
    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))
    def lower(self): return self.__class__(self.data.lower())
    def lstrip(self, chars=None): return self.__class__(self.data.lstrip(chars))
    def partition(self, sep):
        return self.data.partition(sep)
    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))
    def rfind(self, sub, start=0, end=sys.maxint):
        return self.data.rfind(sub, start, end)
    def rindex(self, sub, start=0, end=sys.maxint):
        return self.data.rindex(sub, start, end)
    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))
    def rpartition(self, sep):
        return self.data.rpartition(sep)
    def rstrip(self, chars=None): return self.__class__(self.data.rstrip(chars))
    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)
    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)
    def splitlines(self, keepends=0): return self.data.splitlines(keepends)
    def startswith(self, prefix, start=0, end=sys.maxint):
        return self.data.startswith(prefix, start, end)
    def strip(self, chars=None): return self.__class__(self.data.strip(chars))
    def swapcase(self): return self.__class__(self.data.swapcase())
    def title(self): return self.__class__(self.data.title())
    def translate(self, *args):
        return self.__class__(self.data.translate(*args))
    def upper(self): return self.__class__(self.data.upper())
    def zfill(self, width): return self.__class__(self.data.zfill(width))

class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""
    def __init__(self, string=""):
        self.data = string
    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")
    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + sub + self.data[index+1:]
    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + self.data[index+1:]
    def __setslice__(self, start, end, sub):
        start = max(start, 0); end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start]+sub.data+self.data[end:]
        elif isinstance(sub, basestring):
            self.data = self.data[:start]+sub+self.data[end:]
        else:
            self.data =  self.data[:start]+str(sub)+self.data[end:]
    def __delslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]
    def immutable(self):
        return UserString(self.data)
    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, basestring):
            self.data += other
        else:
            self.data += str(other)
        return self
    def __imul__(self, n):
        self.data *= n
        return self

class String(MutableString, Union):

    _fields_ = [('raw', POINTER(c_char)),
                ('data', c_char_p)]

    def __init__(self, obj=""):
        if isinstance(obj, (str, unicode, UserString)):
            self.data = str(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(POINTER(c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj)

        # Convert from c_char_p
        elif isinstance(obj, c_char_p):
            return obj

        # Convert from POINTER(c_char)
        elif isinstance(obj, POINTER(c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(cast(obj, POINTER(c_char)))

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)
    from_param = classmethod(from_param)

def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)

# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to c_void_p.
def UNCHECKED(type):
    if (hasattr(type, "_type_") and isinstance(type._type_, str)
        and type._type_ != "P"):
        return type
    else:
        return c_void_p

# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self,func,restype,argtypes):
        self.func=func
        self.func.restype=restype
        self.argtypes=argtypes
    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func
    def __call__(self,*args):
        fixed_args=[]
        i=0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i+=1
        return self.func(*fixed_args+list(args[i:]))

# End preamble

_libs = {}
_libdirs = []

# Begin loader

# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import os.path, re, sys, glob
import platform
import ctypes
import ctypes.util

def _environ_path(name):
    if name in os.environ:
        return os.environ[name].split(":")
    else:
        return []

class LibraryLoader(object):
    def __init__(self):
        self.other_dirs=[]

    def load_library(self,libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            if os.path.exists(path):
                return self.load(path)

        raise ImportError("%s not found." % libname)

    def load(self,path):
        """Given a path to a library, load it."""
        try:
            # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
            # of the default RTLD_LOCAL.  Without this, you end up with
            # libraries not being loadable, resulting in "Symbol not found"
            # errors
            if sys.platform == 'darwin':
                return ctypes.CDLL(path, ctypes.RTLD_GLOBAL)
            else:
                return ctypes.cdll.LoadLibrary(path)
        except OSError,e:
            raise ImportError(e)

    def getpaths(self,libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # FIXME / TODO return '.' and os.path.dirname(__file__)
            for path in self.getplatformpaths(libname):
                yield path

            path = ctypes.util.find_library(libname)
            if path: yield path

    def getplatformpaths(self, libname):
        return []

# Darwin (Mac OS X)

class DarwinLibraryLoader(LibraryLoader):
    name_formats = ["lib%s.dylib", "lib%s.so", "lib%s.bundle", "%s.dylib",
                "%s.so", "%s.bundle", "%s"]

    def getplatformpaths(self,libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [format % libname for format in self.name_formats]

        for dir in self.getdirs(libname):
            for name in names:
                yield os.path.join(dir,name)

    def getdirs(self,libname):
        '''Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        '''

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [os.path.expanduser('~/lib'),
                                          '/usr/local/lib', '/usr/lib']

        dirs = []

        if '/' in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))

        dirs.extend(self.other_dirs)
        dirs.append(".")
        dirs.append(os.path.dirname(__file__))

        if hasattr(sys, 'frozen') and sys.frozen == 'macosx_app':
            dirs.append(os.path.join(
                os.environ['RESOURCEPATH'],
                '..',
                'Frameworks'))

        dirs.extend(dyld_fallback_library_path)

        return dirs

# Posix

class PosixLibraryLoader(LibraryLoader):
    _ld_so_cache = None

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = []
        for name in ("LD_LIBRARY_PATH",
                     "SHLIB_PATH", # HPUX
                     "LIBPATH", # OS/2, AIX
                     "LIBRARY_PATH", # BE/OS
                    ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))
        directories.extend(self.other_dirs)
        directories.append(".")
        directories.append(os.path.dirname(__file__))

        try: directories.extend([dir.strip() for dir in open('/etc/ld.so.conf')])
        except IOError: pass

        unix_lib_dirs_list = ['/lib', '/usr/lib', '/lib64', '/usr/lib64']
        if sys.platform.startswith('linux'):
            # Try and support multiarch work in Ubuntu
            # https://wiki.ubuntu.com/MultiarchSpec
            bitage = platform.architecture()[0]
            if bitage.startswith('32'):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ['/lib/i386-linux-gnu', '/usr/lib/i386-linux-gnu']
            elif bitage.startswith('64'):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ['/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu']
            else:
                # guess...
                unix_lib_dirs_list += glob.glob('/lib/*linux-gnu')
        directories.extend(unix_lib_dirs_list)

        cache = {}
        lib_re = re.compile(r'lib(.*)\.s[ol]')
        ext_re = re.compile(r'\.s[ol]$')
        for dir in directories:
            try:
                for path in glob.glob("%s/*.s[ol]*" % dir):
                    file = os.path.basename(path)

                    # Index by filename
                    if file not in cache:
                        cache[file] = path

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        if library not in cache:
                            cache[library] = path
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname)
        if result: yield result

        path = ctypes.util.find_library(libname)
        if path: yield os.path.join("/lib",path)

# Windows

class _WindowsLibrary(object):
    def __init__(self, path):
        self.cdll = ctypes.cdll.LoadLibrary(path)
        self.windll = ctypes.windll.LoadLibrary(path)

    def __getattr__(self, name):
        try: return getattr(self.cdll,name)
        except AttributeError:
            try: return getattr(self.windll,name)
            except AttributeError:
                raise

class WindowsLibraryLoader(LibraryLoader):
    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll"]

    def load_library(self, libname):
        try:
            result = LibraryLoader.load_library(self, libname)
        except ImportError:
            result = None
            if os.path.sep not in libname:
                for name in self.name_formats:
                    try:
                        result = getattr(ctypes.cdll, name % libname)
                        if result:
                            break
                    except WindowsError:
                        result = None
            if result is None:
                try:
                    result = getattr(ctypes.cdll, libname)
                except WindowsError:
                    result = None
            if result is None:
                raise ImportError("%s not found." % libname)
        return result

    def load(self, path):
        return _WindowsLibrary(path)

    def getplatformpaths(self, libname):
        if os.path.sep not in libname:
            for name in self.name_formats:
                dll_in_current_dir = os.path.abspath(name % libname)
                if os.path.exists(dll_in_current_dir):
                    yield dll_in_current_dir
                path = ctypes.util.find_library(name % libname)
                if path:
                    yield path

# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin":   DarwinLibraryLoader,
    "cygwin":   WindowsLibraryLoader,
    "win32":    WindowsLibraryLoader
}

loader = loaderclass.get(sys.platform, PosixLibraryLoader)()

def add_library_search_dirs(other_dirs):
    loader.other_dirs = other_dirs

load_library = loader.load_library

del loaderclass

# End loader

add_library_search_dirs([])

# Begin libraries

_libs["libGPI2.so"] = load_library("libGPI2.so")

# 1 libraries
# End libraries

# No modules

gaspi_char = c_char # GASPI.h: 56

gaspi_uchar = c_ubyte # GASPI.h: 57

gaspi_short = c_short # GASPI.h: 58

gaspi_ushort = c_ushort # GASPI.h: 59

gaspi_int = c_int # GASPI.h: 60

gaspi_uint = c_uint # GASPI.h: 61

gaspi_long = c_long # GASPI.h: 62

gaspi_ulong = c_ulong # GASPI.h: 63

gaspi_float = c_float # GASPI.h: 64

gaspi_double = c_double # GASPI.h: 65

gaspi_timeout_t = c_uint # GASPI.h: 67

gaspi_rank_t = c_ushort # GASPI.h: 68

gaspi_group_t = c_ubyte # GASPI.h: 69

gaspi_number_t = c_uint # GASPI.h: 70

gaspi_pointer_t = POINTER(None) # GASPI.h: 71

gaspi_state_t = POINTER(None) # GASPI.h: 72

gaspi_state_vector_t = POINTER(c_ubyte) # GASPI.h: 73

gaspi_queue_id_t = c_ubyte # GASPI.h: 74

gaspi_size_t = c_ulong # GASPI.h: 75

gaspi_alloc_t = c_ulong # GASPI.h: 76

gaspi_segment_id_t = c_ubyte # GASPI.h: 77

gaspi_offset_t = c_ulong # GASPI.h: 78

gaspi_atomic_value_t = c_ulong # GASPI.h: 79

gaspi_time_t = c_ulong # GASPI.h: 80

gaspi_notification_id_t = c_ushort # GASPI.h: 81

gaspi_notification_t = c_uint # GASPI.h: 82

gaspi_statistic_counter_t = c_uint # GASPI.h: 83

gaspi_string_t = String # GASPI.h: 84

enum_anon_1 = c_int # GASPI.h: 96

GASPI_ERROR = (-1) # GASPI.h: 96

GASPI_SUCCESS = 0 # GASPI.h: 96

GASPI_TIMEOUT = 1 # GASPI.h: 96

gaspi_return_t = enum_anon_1 # GASPI.h: 96

enum_anon_2 = c_int # GASPI.h: 108

GASPI_IB = 0 # GASPI.h: 108

GASPI_ETHERNET = 1 # GASPI.h: 108

GASPI_GEMINI = 2 # GASPI.h: 108

GASPI_ARIES = 3 # GASPI.h: 108

gaspi_network_t = enum_anon_2 # GASPI.h: 108

enum_anon_3 = c_int # GASPI.h: 119

GASPI_OP_MIN = 0 # GASPI.h: 119

GASPI_OP_MAX = 1 # GASPI.h: 119

GASPI_OP_SUM = 2 # GASPI.h: 119

gaspi_operation_t = enum_anon_3 # GASPI.h: 119

enum_anon_4 = c_int # GASPI.h: 133

GASPI_TYPE_INT = 0 # GASPI.h: 133

GASPI_TYPE_UINT = 1 # GASPI.h: 133

GASPI_TYPE_FLOAT = 2 # GASPI.h: 133

GASPI_TYPE_DOUBLE = 3 # GASPI.h: 133

GASPI_TYPE_LONG = 4 # GASPI.h: 133

GASPI_TYPE_ULONG = 5 # GASPI.h: 133

gaspi_datatype_t = enum_anon_4 # GASPI.h: 133

enum_anon_5 = c_int # GASPI.h: 143

GASPI_STATE_HEALTHY = 0 # GASPI.h: 143

GASPI_STATE_CORRUPT = 1 # GASPI.h: 143

gaspi_qp_state_t = enum_anon_5 # GASPI.h: 143

enum_gaspi_alloc_policy_flags = c_int # GASPI.h: 149

GASPI_MEM_UNINITIALIZED = 0 # GASPI.h: 149

GASPI_MEM_INITIALIZED = 1 # GASPI.h: 149

# GASPI.h: 182
class struct_gaspi_config(Structure):
    pass

struct_gaspi_config.__slots__ = [
    'logger',
    'net_info',
    'netdev_id',
    'mtu',
    'port_check',
    'user_net',
    'net_typ',
    'queue_depth',
    'qp_count',
    'group_max',
    'segment_max',
    'transfer_size_max',
    'notification_num',
    'passive_queue_size_max',
    'passive_transfer_size_max',
    'allreduce_buf_size',
    'allreduce_elem_max',
    'build_infrastructure',
]
struct_gaspi_config._fields_ = [
    ('logger', gaspi_uint),
    ('net_info', gaspi_uint),
    ('netdev_id', gaspi_int),
    ('mtu', gaspi_uint),
    ('port_check', gaspi_uint),
    ('user_net', gaspi_uint),
    ('net_typ', gaspi_network_t),
    ('queue_depth', gaspi_uint),
    ('qp_count', gaspi_uint),
    ('group_max', gaspi_number_t),
    ('segment_max', gaspi_number_t),
    ('transfer_size_max', gaspi_size_t),
    ('notification_num', gaspi_number_t),
    ('passive_queue_size_max', gaspi_number_t),
    ('passive_transfer_size_max', gaspi_number_t),
    ('allreduce_buf_size', gaspi_size_t),
    ('allreduce_elem_max', gaspi_number_t),
    ('build_infrastructure', gaspi_number_t),
]

gaspi_config_t = struct_gaspi_config # GASPI.h: 182

enum_anon_6 = c_int # GASPI.h: 191

GASPI_STATISTIC_ARGUMENT_NONE = 0 # GASPI.h: 191

gaspi_statistic_argument_t = enum_anon_6 # GASPI.h: 191

gaspi_reduce_operation_t = CFUNCTYPE(UNCHECKED(gaspi_return_t), gaspi_pointer_t, gaspi_pointer_t, gaspi_pointer_t, gaspi_state_t, gaspi_number_t, gaspi_size_t, gaspi_timeout_t) # GASPI.h: 196

# GASPI.h: 218
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_config_get'):
        continue
    gaspi_config_get = _lib.gaspi_config_get
    gaspi_config_get.argtypes = [POINTER(gaspi_config_t)]
    gaspi_config_get.restype = gaspi_return_t
    break

# GASPI.h: 227
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_config_set'):
        continue
    gaspi_config_set = _lib.gaspi_config_set
    gaspi_config_set.argtypes = [gaspi_config_t]
    gaspi_config_set.restype = gaspi_return_t
    break

# GASPI.h: 236
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_version'):
        continue
    gaspi_version = _lib.gaspi_version
    gaspi_version.argtypes = [POINTER(c_float)]
    gaspi_version.restype = gaspi_return_t
    break

# GASPI.h: 246
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_proc_init'):
        continue
    gaspi_proc_init = _lib.gaspi_proc_init
    gaspi_proc_init.argtypes = [gaspi_timeout_t]
    gaspi_proc_init.restype = gaspi_return_t
    break

# GASPI.h: 257
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_proc_term'):
        continue
    gaspi_proc_term = _lib.gaspi_proc_term
    gaspi_proc_term.argtypes = [gaspi_timeout_t]
    gaspi_proc_term.restype = gaspi_return_t
    break

# GASPI.h: 266
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_proc_rank'):
        continue
    gaspi_proc_rank = _lib.gaspi_proc_rank
    gaspi_proc_rank.argtypes = [POINTER(gaspi_rank_t)]
    gaspi_proc_rank.restype = gaspi_return_t
    break

# GASPI.h: 276
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_proc_num'):
        continue
    gaspi_proc_num = _lib.gaspi_proc_num
    gaspi_proc_num.argtypes = [POINTER(gaspi_rank_t)]
    gaspi_proc_num.restype = gaspi_return_t
    break

# GASPI.h: 287
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_proc_kill'):
        continue
    gaspi_proc_kill = _lib.gaspi_proc_kill
    gaspi_proc_kill.argtypes = [gaspi_rank_t, gaspi_timeout_t]
    gaspi_proc_kill.restype = gaspi_return_t
    break

# GASPI.h: 299
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_connect'):
        continue
    gaspi_connect = _lib.gaspi_connect
    gaspi_connect.argtypes = [gaspi_rank_t, gaspi_timeout_t]
    gaspi_connect.restype = gaspi_return_t
    break

# GASPI.h: 311
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_disconnect'):
        continue
    gaspi_disconnect = _lib.gaspi_disconnect
    gaspi_disconnect.argtypes = [gaspi_rank_t, gaspi_timeout_t]
    gaspi_disconnect.restype = gaspi_return_t
    break

# GASPI.h: 321
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_group_create'):
        continue
    gaspi_group_create = _lib.gaspi_group_create
    gaspi_group_create.argtypes = [POINTER(gaspi_group_t)]
    gaspi_group_create.restype = gaspi_return_t
    break

# GASPI.h: 330
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_group_delete'):
        continue
    gaspi_group_delete = _lib.gaspi_group_delete
    gaspi_group_delete.argtypes = [gaspi_group_t]
    gaspi_group_delete.restype = gaspi_return_t
    break

# GASPI.h: 340
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_group_add'):
        continue
    gaspi_group_add = _lib.gaspi_group_add
    gaspi_group_add.argtypes = [gaspi_group_t, gaspi_rank_t]
    gaspi_group_add.restype = gaspi_return_t
    break

# GASPI.h: 352
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_group_commit'):
        continue
    gaspi_group_commit = _lib.gaspi_group_commit
    gaspi_group_commit.argtypes = [gaspi_group_t, gaspi_timeout_t]
    gaspi_group_commit.restype = gaspi_return_t
    break

# GASPI.h: 362
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_group_num'):
        continue
    gaspi_group_num = _lib.gaspi_group_num
    gaspi_group_num.argtypes = [POINTER(gaspi_number_t)]
    gaspi_group_num.restype = gaspi_return_t
    break

# GASPI.h: 373
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_group_size'):
        continue
    gaspi_group_size = _lib.gaspi_group_size
    gaspi_group_size.argtypes = [gaspi_group_t, POINTER(gaspi_number_t)]
    gaspi_group_size.restype = gaspi_return_t
    break

# GASPI.h: 384
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_group_ranks'):
        continue
    gaspi_group_ranks = _lib.gaspi_group_ranks
    gaspi_group_ranks.argtypes = [gaspi_group_t, POINTER(gaspi_rank_t)]
    gaspi_group_ranks.restype = gaspi_return_t
    break

# GASPI.h: 394
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_group_max'):
        continue
    gaspi_group_max = _lib.gaspi_group_max
    gaspi_group_max.argtypes = [POINTER(gaspi_number_t)]
    gaspi_group_max.restype = gaspi_return_t
    break

# GASPI.h: 405
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_alloc'):
        continue
    gaspi_segment_alloc = _lib.gaspi_segment_alloc
    gaspi_segment_alloc.argtypes = [gaspi_segment_id_t, gaspi_size_t, gaspi_alloc_t]
    gaspi_segment_alloc.restype = gaspi_return_t
    break

# GASPI.h: 415
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_delete'):
        continue
    gaspi_segment_delete = _lib.gaspi_segment_delete
    gaspi_segment_delete.argtypes = [gaspi_segment_id_t]
    gaspi_segment_delete.restype = gaspi_return_t
    break

# GASPI.h: 428
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_register'):
        continue
    gaspi_segment_register = _lib.gaspi_segment_register
    gaspi_segment_register.argtypes = [gaspi_segment_id_t, gaspi_rank_t, gaspi_timeout_t]
    gaspi_segment_register.restype = gaspi_return_t
    break

# GASPI.h: 446
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_create'):
        continue
    gaspi_segment_create = _lib.gaspi_segment_create
    gaspi_segment_create.argtypes = [gaspi_segment_id_t, gaspi_size_t, gaspi_group_t, gaspi_timeout_t, gaspi_alloc_t]
    gaspi_segment_create.restype = gaspi_return_t
    break

# GASPI.h: 459
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_num'):
        continue
    gaspi_segment_num = _lib.gaspi_segment_num
    gaspi_segment_num.argtypes = [POINTER(gaspi_number_t)]
    gaspi_segment_num.restype = gaspi_return_t
    break

# GASPI.h: 469
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_list'):
        continue
    gaspi_segment_list = _lib.gaspi_segment_list
    gaspi_segment_list.argtypes = [gaspi_number_t, POINTER(gaspi_segment_id_t)]
    gaspi_segment_list.restype = gaspi_return_t
    break

# GASPI.h: 481
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_ptr'):
        continue
    gaspi_segment_ptr = _lib.gaspi_segment_ptr
    gaspi_segment_ptr.argtypes = [gaspi_segment_id_t, POINTER(gaspi_pointer_t)]
    gaspi_segment_ptr.restype = gaspi_return_t
    break

# GASPI.h: 493
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_size'):
        continue
    gaspi_segment_size = _lib.gaspi_segment_size
    gaspi_segment_size.argtypes = [gaspi_segment_id_t, gaspi_rank_t, POINTER(gaspi_size_t)]
    gaspi_segment_size.restype = gaspi_return_t
    break

# GASPI.h: 504
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_segment_max'):
        continue
    gaspi_segment_max = _lib.gaspi_segment_max
    gaspi_segment_max.argtypes = [POINTER(gaspi_number_t)]
    gaspi_segment_max.restype = gaspi_return_t
    break

# GASPI.h: 523
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_write'):
        continue
    gaspi_write = _lib.gaspi_write
    gaspi_write.argtypes = [gaspi_segment_id_t, gaspi_offset_t, gaspi_rank_t, gaspi_segment_id_t, gaspi_offset_t, gaspi_size_t, gaspi_queue_id_t, gaspi_timeout_t]
    gaspi_write.restype = gaspi_return_t
    break

# GASPI.h: 547
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_read'):
        continue
    gaspi_read = _lib.gaspi_read
    gaspi_read.argtypes = [gaspi_segment_id_t, gaspi_offset_t, gaspi_rank_t, gaspi_segment_id_t, gaspi_offset_t, gaspi_size_t, gaspi_queue_id_t, gaspi_timeout_t]
    gaspi_read.restype = gaspi_return_t
    break

# GASPI.h: 572
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_write_list'):
        continue
    gaspi_write_list = _lib.gaspi_write_list
    gaspi_write_list.argtypes = [gaspi_number_t, POINTER(gaspi_segment_id_t), POINTER(gaspi_offset_t), gaspi_rank_t, POINTER(gaspi_segment_id_t), POINTER(gaspi_offset_t), POINTER(gaspi_size_t), gaspi_queue_id_t, gaspi_timeout_t]
    gaspi_write_list.restype = gaspi_return_t
    break

# GASPI.h: 601
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_read_list'):
        continue
    gaspi_read_list = _lib.gaspi_read_list
    gaspi_read_list.argtypes = [gaspi_number_t, POINTER(gaspi_segment_id_t), POINTER(gaspi_offset_t), gaspi_rank_t, POINTER(gaspi_segment_id_t), POINTER(gaspi_offset_t), POINTER(gaspi_size_t), gaspi_queue_id_t, gaspi_timeout_t]
    gaspi_read_list.restype = gaspi_return_t
    break

# GASPI.h: 620
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_wait'):
        continue
    gaspi_wait = _lib.gaspi_wait
    gaspi_wait.argtypes = [gaspi_queue_id_t, gaspi_timeout_t]
    gaspi_wait.restype = gaspi_return_t
    break

# GASPI.h: 634
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_barrier'):
        continue
    gaspi_barrier = _lib.gaspi_barrier
    gaspi_barrier.argtypes = [gaspi_group_t, gaspi_timeout_t]
    gaspi_barrier.restype = gaspi_return_t
    break

# GASPI.h: 651
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_allreduce'):
        continue
    gaspi_allreduce = _lib.gaspi_allreduce
    gaspi_allreduce.argtypes = [gaspi_pointer_t, gaspi_pointer_t, gaspi_number_t, gaspi_operation_t, gaspi_datatype_t, gaspi_group_t, gaspi_timeout_t]
    gaspi_allreduce.restype = gaspi_return_t
    break

# GASPI.h: 659
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_allreduce_user'):
        continue
    gaspi_allreduce_user = _lib.gaspi_allreduce_user
    gaspi_allreduce_user.argtypes = [gaspi_pointer_t, gaspi_pointer_t, gaspi_number_t, gaspi_size_t, gaspi_reduce_operation_t, gaspi_state_t, gaspi_group_t, gaspi_timeout_t]
    gaspi_allreduce_user.restype = gaspi_return_t
    break

# GASPI.h: 685
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_atomic_fetch_add'):
        continue
    gaspi_atomic_fetch_add = _lib.gaspi_atomic_fetch_add
    gaspi_atomic_fetch_add.argtypes = [gaspi_segment_id_t, gaspi_offset_t, gaspi_rank_t, gaspi_atomic_value_t, POINTER(gaspi_atomic_value_t), gaspi_timeout_t]
    gaspi_atomic_fetch_add.restype = gaspi_return_t
    break

# GASPI.h: 705
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_atomic_compare_swap'):
        continue
    gaspi_atomic_compare_swap = _lib.gaspi_atomic_compare_swap
    gaspi_atomic_compare_swap.argtypes = [gaspi_segment_id_t, gaspi_offset_t, gaspi_rank_t, gaspi_atomic_value_t, gaspi_atomic_value_t, POINTER(gaspi_atomic_value_t), gaspi_timeout_t]
    gaspi_atomic_compare_swap.restype = gaspi_return_t
    break

# GASPI.h: 732
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_passive_send'):
        continue
    gaspi_passive_send = _lib.gaspi_passive_send
    gaspi_passive_send.argtypes = [gaspi_segment_id_t, gaspi_offset_t, gaspi_rank_t, gaspi_size_t, gaspi_timeout_t]
    gaspi_passive_send.restype = gaspi_return_t
    break

# GASPI.h: 751
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_passive_receive'):
        continue
    gaspi_passive_receive = _lib.gaspi_passive_receive
    gaspi_passive_receive.argtypes = [gaspi_segment_id_t, gaspi_offset_t, POINTER(gaspi_rank_t), gaspi_size_t, gaspi_timeout_t]
    gaspi_passive_receive.restype = gaspi_return_t
    break

# GASPI.h: 774
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_notify'):
        continue
    gaspi_notify = _lib.gaspi_notify
    gaspi_notify.argtypes = [gaspi_segment_id_t, gaspi_rank_t, gaspi_notification_id_t, gaspi_notification_t, gaspi_queue_id_t, gaspi_timeout_t]
    gaspi_notify.restype = gaspi_return_t
    break

# GASPI.h: 793
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_notify_waitsome'):
        continue
    gaspi_notify_waitsome = _lib.gaspi_notify_waitsome
    gaspi_notify_waitsome.argtypes = [gaspi_segment_id_t, gaspi_notification_id_t, gaspi_number_t, POINTER(gaspi_notification_id_t), gaspi_timeout_t]
    gaspi_notify_waitsome.restype = gaspi_return_t
    break

# GASPI.h: 811
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_notify_reset'):
        continue
    gaspi_notify_reset = _lib.gaspi_notify_reset
    gaspi_notify_reset.argtypes = [gaspi_segment_id_t, gaspi_notification_id_t, POINTER(gaspi_notification_t)]
    gaspi_notify_reset.restype = gaspi_return_t
    break

# GASPI.h: 835
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_write_notify'):
        continue
    gaspi_write_notify = _lib.gaspi_write_notify
    gaspi_write_notify.argtypes = [gaspi_segment_id_t, gaspi_offset_t, gaspi_rank_t, gaspi_segment_id_t, gaspi_offset_t, gaspi_size_t, gaspi_notification_id_t, gaspi_notification_t, gaspi_queue_id_t, gaspi_timeout_t]
    gaspi_write_notify.restype = gaspi_return_t
    break

# GASPI.h: 869
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_write_list_notify'):
        continue
    gaspi_write_list_notify = _lib.gaspi_write_list_notify
    gaspi_write_list_notify.argtypes = [gaspi_number_t, POINTER(gaspi_segment_id_t), POINTER(gaspi_offset_t), gaspi_rank_t, POINTER(gaspi_segment_id_t), POINTER(gaspi_offset_t), POINTER(gaspi_size_t), gaspi_segment_id_t, gaspi_notification_id_t, gaspi_notification_t, gaspi_queue_id_t, gaspi_timeout_t]
    gaspi_write_list_notify.restype = gaspi_return_t
    break

# GASPI.h: 900
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_queue_size'):
        continue
    gaspi_queue_size = _lib.gaspi_queue_size
    gaspi_queue_size.argtypes = [gaspi_queue_id_t, POINTER(gaspi_number_t)]
    gaspi_queue_size.restype = gaspi_return_t
    break

# GASPI.h: 910
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_queue_num'):
        continue
    gaspi_queue_num = _lib.gaspi_queue_num
    gaspi_queue_num.argtypes = [POINTER(gaspi_number_t)]
    gaspi_queue_num.restype = gaspi_return_t
    break

# GASPI.h: 921
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_queue_size_max'):
        continue
    gaspi_queue_size_max = _lib.gaspi_queue_size_max
    gaspi_queue_size_max.argtypes = [POINTER(gaspi_number_t)]
    gaspi_queue_size_max.restype = gaspi_return_t
    break

# GASPI.h: 932
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_transfer_size_min'):
        continue
    gaspi_transfer_size_min = _lib.gaspi_transfer_size_min
    gaspi_transfer_size_min.argtypes = [POINTER(gaspi_size_t)]
    gaspi_transfer_size_min.restype = gaspi_return_t
    break

# GASPI.h: 943
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_transfer_size_max'):
        continue
    gaspi_transfer_size_max = _lib.gaspi_transfer_size_max
    gaspi_transfer_size_max.argtypes = [POINTER(gaspi_size_t)]
    gaspi_transfer_size_max.restype = gaspi_return_t
    break

# GASPI.h: 954
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_notification_num'):
        continue
    gaspi_notification_num = _lib.gaspi_notification_num
    gaspi_notification_num.argtypes = [POINTER(gaspi_number_t)]
    gaspi_notification_num.restype = gaspi_return_t
    break

# GASPI.h: 964
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_passive_transfer_size_max'):
        continue
    gaspi_passive_transfer_size_max = _lib.gaspi_passive_transfer_size_max
    gaspi_passive_transfer_size_max.argtypes = [POINTER(gaspi_size_t)]
    gaspi_passive_transfer_size_max.restype = gaspi_return_t
    break

# GASPI.h: 975
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_allreduce_buf_size'):
        continue
    gaspi_allreduce_buf_size = _lib.gaspi_allreduce_buf_size
    gaspi_allreduce_buf_size.argtypes = [POINTER(gaspi_size_t)]
    gaspi_allreduce_buf_size.restype = gaspi_return_t
    break

# GASPI.h: 984
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_allreduce_elem_max'):
        continue
    gaspi_allreduce_elem_max = _lib.gaspi_allreduce_elem_max
    gaspi_allreduce_elem_max.argtypes = [POINTER(gaspi_number_t)]
    gaspi_allreduce_elem_max.restype = gaspi_return_t
    break

# GASPI.h: 994
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_rw_list_elem_max'):
        continue
    gaspi_rw_list_elem_max = _lib.gaspi_rw_list_elem_max
    gaspi_rw_list_elem_max.argtypes = [POINTER(gaspi_number_t)]
    gaspi_rw_list_elem_max.restype = gaspi_return_t
    break

# GASPI.h: 1003
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_network_type'):
        continue
    gaspi_network_type = _lib.gaspi_network_type
    gaspi_network_type.argtypes = [POINTER(gaspi_network_t)]
    gaspi_network_type.restype = gaspi_return_t
    break

# GASPI.h: 1012
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_time_ticks'):
        continue
    gaspi_time_ticks = _lib.gaspi_time_ticks
    gaspi_time_ticks.argtypes = [POINTER(gaspi_time_t)]
    gaspi_time_ticks.restype = gaspi_return_t
    break

# GASPI.h: 1021
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_cpu_frequency'):
        continue
    gaspi_cpu_frequency = _lib.gaspi_cpu_frequency
    gaspi_cpu_frequency.argtypes = [POINTER(gaspi_float)]
    gaspi_cpu_frequency.restype = gaspi_return_t
    break

# GASPI.h: 1030
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_machine_type'):
        continue
    gaspi_machine_type = _lib.gaspi_machine_type
    gaspi_machine_type.argtypes = [c_char * 16]
    gaspi_machine_type.restype = gaspi_return_t
    break

# GASPI.h: 1040
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_state_vec_get'):
        continue
    gaspi_state_vec_get = _lib.gaspi_state_vec_get
    gaspi_state_vec_get.argtypes = [gaspi_state_vector_t]
    gaspi_state_vec_get.restype = gaspi_return_t
    break

# GASPI.h: 1048
for _lib in _libs.values():
    if hasattr(_lib, 'gaspi_printf'):
        _func = _lib.gaspi_printf
        _restype = None
        _argtypes = [String]
        gaspi_printf = _variadic_function(_func,_restype,_argtypes)

# GASPI.h: 1054
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_print_affinity_mask'):
        continue
    gaspi_print_affinity_mask = _lib.gaspi_print_affinity_mask
    gaspi_print_affinity_mask.argtypes = []
    gaspi_print_affinity_mask.restype = None
    break

# GASPI.h: 1060
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_set_socket_affinity'):
        continue
    gaspi_set_socket_affinity = _lib.gaspi_set_socket_affinity
    gaspi_set_socket_affinity.argtypes = [gaspi_uchar]
    gaspi_set_socket_affinity.restype = gaspi_return_t
    break

# GASPI.h: 1072
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_statistic_verbosity_level'):
        continue
    gaspi_statistic_verbosity_level = _lib.gaspi_statistic_verbosity_level
    gaspi_statistic_verbosity_level.argtypes = [gaspi_number_t]
    gaspi_statistic_verbosity_level.restype = gaspi_return_t
    break

# GASPI.h: 1081
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_statistic_counter_max'):
        continue
    gaspi_statistic_counter_max = _lib.gaspi_statistic_counter_max
    gaspi_statistic_counter_max.argtypes = [POINTER(gaspi_statistic_counter_t)]
    gaspi_statistic_counter_max.restype = gaspi_return_t
    break

# GASPI.h: 1095
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_statistic_counter_info'):
        continue
    gaspi_statistic_counter_info = _lib.gaspi_statistic_counter_info
    gaspi_statistic_counter_info.argtypes = [gaspi_statistic_counter_t, POINTER(gaspi_statistic_argument_t), POINTER(gaspi_string_t), POINTER(gaspi_string_t), POINTER(gaspi_number_t)]
    gaspi_statistic_counter_info.restype = gaspi_return_t
    break

# GASPI.h: 1112
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_statistic_counter_get'):
        continue
    gaspi_statistic_counter_get = _lib.gaspi_statistic_counter_get
    gaspi_statistic_counter_get.argtypes = [gaspi_statistic_counter_t, gaspi_number_t, POINTER(gaspi_number_t)]
    gaspi_statistic_counter_get.restype = gaspi_return_t
    break

# GASPI.h: 1124
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gaspi_statistic_counter_reset'):
        continue
    gaspi_statistic_counter_reset = _lib.gaspi_statistic_counter_reset
    gaspi_statistic_counter_reset.argtypes = [gaspi_statistic_counter_t]
    gaspi_statistic_counter_reset.restype = gaspi_return_t
    break

# GASPI.h: 37
try:
    GASPI_MAJOR_VERSION = 1
except:
    pass

# GASPI.h: 38
try:
    GASPI_MINOR_VERSION = 0
except:
    pass

# GASPI.h: 39
try:
    GASPI_REVISION = 1
except:
    pass

# GASPI.h: 41
try:
    GASPI_BLOCK = 4294967295
except:
    pass

# GASPI.h: 42
try:
    GASPI_TEST = 0
except:
    pass

# GASPI.h: 43
try:
    GASPI_MAX_NODES = 65536
except:
    pass

# GASPI.h: 44
try:
    GASPI_SN_PORT = 10840
except:
    pass

# GASPI.h: 45
try:
    GASPI_MAX_GROUPS = 32
except:
    pass

# GASPI.h: 46
try:
    GASPI_MAX_MSEGS = 32
except:
    pass

# GASPI.h: 47
try:
    GASPI_GROUP_ALL = 0
except:
    pass

# GASPI.h: 48
try:
    GASPI_MAX_QP = 16
except:
    pass

# GASPI.h: 49
try:
    GASPI_COLL_QP = GASPI_MAX_QP
except:
    pass

# GASPI.h: 50
try:
    GASPI_PASSIVE_QP = (GASPI_MAX_QP + 1)
except:
    pass

# GASPI.h: 51
try:
    GASPI_MAX_TSIZE_C = ((1 << 31) - 1)
except:
    pass

# GASPI.h: 52
try:
    GASPI_MAX_TSIZE_P = ((1 << 16) - 1)
except:
    pass

# GASPI.h: 53
try:
    GASPI_MAX_QSIZE = 4096
except:
    pass

# GASPI.h: 54
try:
    GASPI_MAX_NOTIFICATION = 65536
except:
    pass

# GASPI.h: 155
try:
    GASPI_ALLOC_DEFAULT = GASPI_MEM_UNINITIALIZED
except:
    pass

gaspi_config = struct_gaspi_config # GASPI.h: 182

# No inserted files

