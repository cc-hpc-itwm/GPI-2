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

_libs["libibverbs.so.1"] = load_library("libibverbs.so.1")

# 1 libraries
# End libraries

# No modules

# /usr/include/bits/pthreadtypes.h: 61
class struct___pthread_internal_list(Structure):
    pass

struct___pthread_internal_list.__slots__ = [
    '__prev',
    '__next',
]
struct___pthread_internal_list._fields_ = [
    ('__prev', POINTER(struct___pthread_internal_list)),
    ('__next', POINTER(struct___pthread_internal_list)),
]

__pthread_list_t = struct___pthread_internal_list # /usr/include/bits/pthreadtypes.h: 65

# /usr/include/bits/pthreadtypes.h: 78
class struct___pthread_mutex_s(Structure):
    pass

struct___pthread_mutex_s.__slots__ = [
    '__lock',
    '__count',
    '__owner',
    '__nusers',
    '__kind',
    '__spins',
    '__list',
]
struct___pthread_mutex_s._fields_ = [
    ('__lock', c_int),
    ('__count', c_uint),
    ('__owner', c_int),
    ('__nusers', c_uint),
    ('__kind', c_int),
    ('__spins', c_int),
    ('__list', __pthread_list_t),
]

# /usr/include/bits/pthreadtypes.h: 104
class union_anon_4(Union):
    pass

union_anon_4.__slots__ = [
    '__data',
    '__size',
    '__align',
]
union_anon_4._fields_ = [
    ('__data', struct___pthread_mutex_s),
    ('__size', c_char * 40),
    ('__align', c_long),
]

pthread_mutex_t = union_anon_4 # /usr/include/bits/pthreadtypes.h: 104

# /usr/include/bits/pthreadtypes.h: 117
class struct_anon_6(Structure):
    pass

struct_anon_6.__slots__ = [
    '__lock',
    '__futex',
    '__total_seq',
    '__wakeup_seq',
    '__woken_seq',
    '__mutex',
    '__nwaiters',
    '__broadcast_seq',
]
struct_anon_6._fields_ = [
    ('__lock', c_int),
    ('__futex', c_uint),
    ('__total_seq', c_ulonglong),
    ('__wakeup_seq', c_ulonglong),
    ('__woken_seq', c_ulonglong),
    ('__mutex', POINTER(None)),
    ('__nwaiters', c_uint),
    ('__broadcast_seq', c_uint),
]

# /usr/include/bits/pthreadtypes.h: 130
class union_anon_7(Union):
    pass

union_anon_7.__slots__ = [
    '__data',
    '__size',
    '__align',
]
union_anon_7._fields_ = [
    ('__data', struct_anon_6),
    ('__size', c_char * 48),
    ('__align', c_longlong),
]

pthread_cond_t = union_anon_7 # /usr/include/bits/pthreadtypes.h: 130

# verbs.h: 60
class struct_anon_25(Structure):
    pass

struct_anon_25.__slots__ = [
    'subnet_prefix',
    'interface_id',
]
struct_anon_25._fields_ = [
    ('subnet_prefix', c_uint64),
    ('interface_id', c_uint64),
]

# verbs.h: 58
class union_ibv_gid(Union):
    pass

union_ibv_gid.__slots__ = [
    'raw',
    '_global',
]
union_ibv_gid._fields_ = [
    ('raw', c_uint8 * 16),
    ('_global', struct_anon_25),
]

enum_ibv_node_type = c_int # verbs.h: 66

IBV_NODE_UNKNOWN = (-1) # verbs.h: 66

IBV_NODE_CA = 1 # verbs.h: 66

IBV_NODE_SWITCH = (IBV_NODE_CA + 1) # verbs.h: 66

IBV_NODE_ROUTER = (IBV_NODE_SWITCH + 1) # verbs.h: 66

IBV_NODE_RNIC = (IBV_NODE_ROUTER + 1) # verbs.h: 66

enum_ibv_transport_type = c_int # verbs.h: 74

IBV_TRANSPORT_UNKNOWN = (-1) # verbs.h: 74

IBV_TRANSPORT_IB = 0 # verbs.h: 74

IBV_TRANSPORT_IWARP = (IBV_TRANSPORT_IB + 1) # verbs.h: 74

enum_ibv_device_cap_flags = c_int # verbs.h: 80

IBV_DEVICE_RESIZE_MAX_WR = 1 # verbs.h: 80

IBV_DEVICE_BAD_PKEY_CNTR = (1 << 1) # verbs.h: 80

IBV_DEVICE_BAD_QKEY_CNTR = (1 << 2) # verbs.h: 80

IBV_DEVICE_RAW_MULTI = (1 << 3) # verbs.h: 80

IBV_DEVICE_AUTO_PATH_MIG = (1 << 4) # verbs.h: 80

IBV_DEVICE_CHANGE_PHY_PORT = (1 << 5) # verbs.h: 80

IBV_DEVICE_UD_AV_PORT_ENFORCE = (1 << 6) # verbs.h: 80

IBV_DEVICE_CURR_QP_STATE_MOD = (1 << 7) # verbs.h: 80

IBV_DEVICE_SHUTDOWN_PORT = (1 << 8) # verbs.h: 80

IBV_DEVICE_INIT_TYPE = (1 << 9) # verbs.h: 80

IBV_DEVICE_PORT_ACTIVE_EVENT = (1 << 10) # verbs.h: 80

IBV_DEVICE_SYS_IMAGE_GUID = (1 << 11) # verbs.h: 80

IBV_DEVICE_RC_RNR_NAK_GEN = (1 << 12) # verbs.h: 80

IBV_DEVICE_SRQ_RESIZE = (1 << 13) # verbs.h: 80

IBV_DEVICE_N_NOTIFY_CQ = (1 << 14) # verbs.h: 80

enum_ibv_atomic_cap = c_int # verbs.h: 98

IBV_ATOMIC_NONE = 0 # verbs.h: 98

IBV_ATOMIC_HCA = (IBV_ATOMIC_NONE + 1) # verbs.h: 98

IBV_ATOMIC_GLOB = (IBV_ATOMIC_HCA + 1) # verbs.h: 98

# verbs.h: 104
class struct_ibv_device_attr(Structure):
    pass

struct_ibv_device_attr.__slots__ = [
    'fw_ver',
    'node_guid',
    'sys_image_guid',
    'max_mr_size',
    'page_size_cap',
    'vendor_id',
    'vendor_part_id',
    'hw_ver',
    'max_qp',
    'max_qp_wr',
    'device_cap_flags',
    'max_sge',
    'max_sge_rd',
    'max_cq',
    'max_cqe',
    'max_mr',
    'max_pd',
    'max_qp_rd_atom',
    'max_ee_rd_atom',
    'max_res_rd_atom',
    'max_qp_init_rd_atom',
    'max_ee_init_rd_atom',
    'atomic_cap',
    'max_ee',
    'max_rdd',
    'max_mw',
    'max_raw_ipv6_qp',
    'max_raw_ethy_qp',
    'max_mcast_grp',
    'max_mcast_qp_attach',
    'max_total_mcast_qp_attach',
    'max_ah',
    'max_fmr',
    'max_map_per_fmr',
    'max_srq',
    'max_srq_wr',
    'max_srq_sge',
    'max_pkeys',
    'local_ca_ack_delay',
    'phys_port_cnt',
]
struct_ibv_device_attr._fields_ = [
    ('fw_ver', c_char * 64),
    ('node_guid', c_uint64),
    ('sys_image_guid', c_uint64),
    ('max_mr_size', c_uint64),
    ('page_size_cap', c_uint64),
    ('vendor_id', c_uint32),
    ('vendor_part_id', c_uint32),
    ('hw_ver', c_uint32),
    ('max_qp', c_int),
    ('max_qp_wr', c_int),
    ('device_cap_flags', c_int),
    ('max_sge', c_int),
    ('max_sge_rd', c_int),
    ('max_cq', c_int),
    ('max_cqe', c_int),
    ('max_mr', c_int),
    ('max_pd', c_int),
    ('max_qp_rd_atom', c_int),
    ('max_ee_rd_atom', c_int),
    ('max_res_rd_atom', c_int),
    ('max_qp_init_rd_atom', c_int),
    ('max_ee_init_rd_atom', c_int),
    ('atomic_cap', enum_ibv_atomic_cap),
    ('max_ee', c_int),
    ('max_rdd', c_int),
    ('max_mw', c_int),
    ('max_raw_ipv6_qp', c_int),
    ('max_raw_ethy_qp', c_int),
    ('max_mcast_grp', c_int),
    ('max_mcast_qp_attach', c_int),
    ('max_total_mcast_qp_attach', c_int),
    ('max_ah', c_int),
    ('max_fmr', c_int),
    ('max_map_per_fmr', c_int),
    ('max_srq', c_int),
    ('max_srq_wr', c_int),
    ('max_srq_sge', c_int),
    ('max_pkeys', c_uint16),
    ('local_ca_ack_delay', c_uint8),
    ('phys_port_cnt', c_uint8),
]

enum_ibv_mtu = c_int # verbs.h: 147

IBV_MTU_256 = 1 # verbs.h: 147

IBV_MTU_512 = 2 # verbs.h: 147

IBV_MTU_1024 = 3 # verbs.h: 147

IBV_MTU_2048 = 4 # verbs.h: 147

IBV_MTU_4096 = 5 # verbs.h: 147

enum_ibv_port_state = c_int # verbs.h: 155

IBV_PORT_NOP = 0 # verbs.h: 155

IBV_PORT_DOWN = 1 # verbs.h: 155

IBV_PORT_INIT = 2 # verbs.h: 155

IBV_PORT_ARMED = 3 # verbs.h: 155

IBV_PORT_ACTIVE = 4 # verbs.h: 155

IBV_PORT_ACTIVE_DEFER = 5 # verbs.h: 155

enum_anon_26 = c_int # verbs.h: 164

IBV_LINK_LAYER_UNSPECIFIED = 0 # verbs.h: 164

IBV_LINK_LAYER_INFINIBAND = (IBV_LINK_LAYER_UNSPECIFIED + 1) # verbs.h: 164

IBV_LINK_LAYER_ETHERNET = (IBV_LINK_LAYER_INFINIBAND + 1) # verbs.h: 164

# verbs.h: 170
class struct_ibv_port_attr(Structure):
    pass

struct_ibv_port_attr.__slots__ = [
    'state',
    'max_mtu',
    'active_mtu',
    'gid_tbl_len',
    'port_cap_flags',
    'max_msg_sz',
    'bad_pkey_cntr',
    'qkey_viol_cntr',
    'pkey_tbl_len',
    'lid',
    'sm_lid',
    'lmc',
    'max_vl_num',
    'sm_sl',
    'subnet_timeout',
    'init_type_reply',
    'active_width',
    'active_speed',
    'phys_state',
    'link_layer',
    'reserved',
]
struct_ibv_port_attr._fields_ = [
    ('state', enum_ibv_port_state),
    ('max_mtu', enum_ibv_mtu),
    ('active_mtu', enum_ibv_mtu),
    ('gid_tbl_len', c_int),
    ('port_cap_flags', c_uint32),
    ('max_msg_sz', c_uint32),
    ('bad_pkey_cntr', c_uint32),
    ('qkey_viol_cntr', c_uint32),
    ('pkey_tbl_len', c_uint16),
    ('lid', c_uint16),
    ('sm_lid', c_uint16),
    ('lmc', c_uint8),
    ('max_vl_num', c_uint8),
    ('sm_sl', c_uint8),
    ('subnet_timeout', c_uint8),
    ('init_type_reply', c_uint8),
    ('active_width', c_uint8),
    ('active_speed', c_uint8),
    ('phys_state', c_uint8),
    ('link_layer', c_uint8),
    ('reserved', c_uint8),
]

enum_ibv_event_type = c_int # verbs.h: 194

IBV_EVENT_CQ_ERR = 0 # verbs.h: 194

IBV_EVENT_QP_FATAL = (IBV_EVENT_CQ_ERR + 1) # verbs.h: 194

IBV_EVENT_QP_REQ_ERR = (IBV_EVENT_QP_FATAL + 1) # verbs.h: 194

IBV_EVENT_QP_ACCESS_ERR = (IBV_EVENT_QP_REQ_ERR + 1) # verbs.h: 194

IBV_EVENT_COMM_EST = (IBV_EVENT_QP_ACCESS_ERR + 1) # verbs.h: 194

IBV_EVENT_SQ_DRAINED = (IBV_EVENT_COMM_EST + 1) # verbs.h: 194

IBV_EVENT_PATH_MIG = (IBV_EVENT_SQ_DRAINED + 1) # verbs.h: 194

IBV_EVENT_PATH_MIG_ERR = (IBV_EVENT_PATH_MIG + 1) # verbs.h: 194

IBV_EVENT_DEVICE_FATAL = (IBV_EVENT_PATH_MIG_ERR + 1) # verbs.h: 194

IBV_EVENT_PORT_ACTIVE = (IBV_EVENT_DEVICE_FATAL + 1) # verbs.h: 194

IBV_EVENT_PORT_ERR = (IBV_EVENT_PORT_ACTIVE + 1) # verbs.h: 194

IBV_EVENT_LID_CHANGE = (IBV_EVENT_PORT_ERR + 1) # verbs.h: 194

IBV_EVENT_PKEY_CHANGE = (IBV_EVENT_LID_CHANGE + 1) # verbs.h: 194

IBV_EVENT_SM_CHANGE = (IBV_EVENT_PKEY_CHANGE + 1) # verbs.h: 194

IBV_EVENT_SRQ_ERR = (IBV_EVENT_SM_CHANGE + 1) # verbs.h: 194

IBV_EVENT_SRQ_LIMIT_REACHED = (IBV_EVENT_SRQ_ERR + 1) # verbs.h: 194

IBV_EVENT_QP_LAST_WQE_REACHED = (IBV_EVENT_SRQ_LIMIT_REACHED + 1) # verbs.h: 194

IBV_EVENT_CLIENT_REREGISTER = (IBV_EVENT_QP_LAST_WQE_REACHED + 1) # verbs.h: 194

IBV_EVENT_GID_CHANGE = (IBV_EVENT_CLIENT_REREGISTER + 1) # verbs.h: 194

# verbs.h: 613
class struct_ibv_cq(Structure):
    pass

# verbs.h: 590
class struct_ibv_qp(Structure):
    pass

# verbs.h: 579
class struct_ibv_srq(Structure):
    pass

# verbs.h: 217
class union_anon_27(Union):
    pass

union_anon_27.__slots__ = [
    'cq',
    'qp',
    'srq',
    'port_num',
]
union_anon_27._fields_ = [
    ('cq', POINTER(struct_ibv_cq)),
    ('qp', POINTER(struct_ibv_qp)),
    ('srq', POINTER(struct_ibv_srq)),
    ('port_num', c_int),
]

# verbs.h: 216
class struct_ibv_async_event(Structure):
    pass

struct_ibv_async_event.__slots__ = [
    'element',
    'event_type',
]
struct_ibv_async_event._fields_ = [
    ('element', union_anon_27),
    ('event_type', enum_ibv_event_type),
]

enum_ibv_wc_status = c_int # verbs.h: 226

IBV_WC_SUCCESS = 0 # verbs.h: 226

IBV_WC_LOC_LEN_ERR = (IBV_WC_SUCCESS + 1) # verbs.h: 226

IBV_WC_LOC_QP_OP_ERR = (IBV_WC_LOC_LEN_ERR + 1) # verbs.h: 226

IBV_WC_LOC_EEC_OP_ERR = (IBV_WC_LOC_QP_OP_ERR + 1) # verbs.h: 226

IBV_WC_LOC_PROT_ERR = (IBV_WC_LOC_EEC_OP_ERR + 1) # verbs.h: 226

IBV_WC_WR_FLUSH_ERR = (IBV_WC_LOC_PROT_ERR + 1) # verbs.h: 226

IBV_WC_MW_BIND_ERR = (IBV_WC_WR_FLUSH_ERR + 1) # verbs.h: 226

IBV_WC_BAD_RESP_ERR = (IBV_WC_MW_BIND_ERR + 1) # verbs.h: 226

IBV_WC_LOC_ACCESS_ERR = (IBV_WC_BAD_RESP_ERR + 1) # verbs.h: 226

IBV_WC_REM_INV_REQ_ERR = (IBV_WC_LOC_ACCESS_ERR + 1) # verbs.h: 226

IBV_WC_REM_ACCESS_ERR = (IBV_WC_REM_INV_REQ_ERR + 1) # verbs.h: 226

IBV_WC_REM_OP_ERR = (IBV_WC_REM_ACCESS_ERR + 1) # verbs.h: 226

IBV_WC_RETRY_EXC_ERR = (IBV_WC_REM_OP_ERR + 1) # verbs.h: 226

IBV_WC_RNR_RETRY_EXC_ERR = (IBV_WC_RETRY_EXC_ERR + 1) # verbs.h: 226

IBV_WC_LOC_RDD_VIOL_ERR = (IBV_WC_RNR_RETRY_EXC_ERR + 1) # verbs.h: 226

IBV_WC_REM_INV_RD_REQ_ERR = (IBV_WC_LOC_RDD_VIOL_ERR + 1) # verbs.h: 226

IBV_WC_REM_ABORT_ERR = (IBV_WC_REM_INV_RD_REQ_ERR + 1) # verbs.h: 226

IBV_WC_INV_EECN_ERR = (IBV_WC_REM_ABORT_ERR + 1) # verbs.h: 226

IBV_WC_INV_EEC_STATE_ERR = (IBV_WC_INV_EECN_ERR + 1) # verbs.h: 226

IBV_WC_FATAL_ERR = (IBV_WC_INV_EEC_STATE_ERR + 1) # verbs.h: 226

IBV_WC_RESP_TIMEOUT_ERR = (IBV_WC_FATAL_ERR + 1) # verbs.h: 226

IBV_WC_GENERAL_ERR = (IBV_WC_RESP_TIMEOUT_ERR + 1) # verbs.h: 226

# verbs.h: 250
if hasattr(_libs['libibverbs.so.1'], 'ibv_wc_status_str'):
    ibv_wc_status_str = _libs['libibverbs.so.1'].ibv_wc_status_str
    ibv_wc_status_str.argtypes = [enum_ibv_wc_status]
    if sizeof(c_int) == sizeof(c_void_p):
        ibv_wc_status_str.restype = ReturnString
    else:
        ibv_wc_status_str.restype = String
        ibv_wc_status_str.errcheck = ReturnString

enum_ibv_wc_opcode = c_int # verbs.h: 252

IBV_WC_SEND = 0 # verbs.h: 252

IBV_WC_RDMA_WRITE = (IBV_WC_SEND + 1) # verbs.h: 252

IBV_WC_RDMA_READ = (IBV_WC_RDMA_WRITE + 1) # verbs.h: 252

IBV_WC_COMP_SWAP = (IBV_WC_RDMA_READ + 1) # verbs.h: 252

IBV_WC_FETCH_ADD = (IBV_WC_COMP_SWAP + 1) # verbs.h: 252

IBV_WC_BIND_MW = (IBV_WC_FETCH_ADD + 1) # verbs.h: 252

IBV_WC_RECV = (1 << 7) # verbs.h: 252

IBV_WC_RECV_RDMA_WITH_IMM = (IBV_WC_RECV + 1) # verbs.h: 252

enum_ibv_wc_flags = c_int # verbs.h: 267

IBV_WC_GRH = (1 << 0) # verbs.h: 267

IBV_WC_WITH_IMM = (1 << 1) # verbs.h: 267

# verbs.h: 272
class struct_ibv_wc(Structure):
    pass

struct_ibv_wc.__slots__ = [
    'wr_id',
    'status',
    'opcode',
    'vendor_err',
    'byte_len',
    'imm_data',
    'qp_num',
    'src_qp',
    'wc_flags',
    'pkey_index',
    'slid',
    'sl',
    'dlid_path_bits',
]
struct_ibv_wc._fields_ = [
    ('wr_id', c_uint64),
    ('status', enum_ibv_wc_status),
    ('opcode', enum_ibv_wc_opcode),
    ('vendor_err', c_uint32),
    ('byte_len', c_uint32),
    ('imm_data', c_uint32),
    ('qp_num', c_uint32),
    ('src_qp', c_uint32),
    ('wc_flags', c_int),
    ('pkey_index', c_uint16),
    ('slid', c_uint16),
    ('sl', c_uint8),
    ('dlid_path_bits', c_uint8),
]

enum_ibv_access_flags = c_int # verbs.h: 288

IBV_ACCESS_LOCAL_WRITE = 1 # verbs.h: 288

IBV_ACCESS_REMOTE_WRITE = (1 << 1) # verbs.h: 288

IBV_ACCESS_REMOTE_READ = (1 << 2) # verbs.h: 288

IBV_ACCESS_REMOTE_ATOMIC = (1 << 3) # verbs.h: 288

IBV_ACCESS_MW_BIND = (1 << 4) # verbs.h: 288

# verbs.h: 717
class struct_ibv_context(Structure):
    pass

# verbs.h: 296
class struct_ibv_pd(Structure):
    pass

struct_ibv_pd.__slots__ = [
    'context',
    'handle',
]
struct_ibv_pd._fields_ = [
    ('context', POINTER(struct_ibv_context)),
    ('handle', c_uint32),
]

enum_ibv_rereg_mr_flags = c_int # verbs.h: 301

IBV_REREG_MR_CHANGE_TRANSLATION = (1 << 0) # verbs.h: 301

IBV_REREG_MR_CHANGE_PD = (1 << 1) # verbs.h: 301

IBV_REREG_MR_CHANGE_ACCESS = (1 << 2) # verbs.h: 301

IBV_REREG_MR_KEEP_VALID = (1 << 3) # verbs.h: 301

# verbs.h: 308
class struct_ibv_mr(Structure):
    pass

struct_ibv_mr.__slots__ = [
    'context',
    'pd',
    'addr',
    'length',
    'handle',
    'lkey',
    'rkey',
]
struct_ibv_mr._fields_ = [
    ('context', POINTER(struct_ibv_context)),
    ('pd', POINTER(struct_ibv_pd)),
    ('addr', POINTER(None)),
    ('length', c_size_t),
    ('handle', c_uint32),
    ('lkey', c_uint32),
    ('rkey', c_uint32),
]

enum_ibv_mw_type = c_int # verbs.h: 318

IBV_MW_TYPE_1 = 1 # verbs.h: 318

IBV_MW_TYPE_2 = 2 # verbs.h: 318

# verbs.h: 323
class struct_ibv_mw(Structure):
    pass

struct_ibv_mw.__slots__ = [
    'context',
    'pd',
    'rkey',
]
struct_ibv_mw._fields_ = [
    ('context', POINTER(struct_ibv_context)),
    ('pd', POINTER(struct_ibv_pd)),
    ('rkey', c_uint32),
]

# verbs.h: 329
class struct_ibv_global_route(Structure):
    pass

struct_ibv_global_route.__slots__ = [
    'dgid',
    'flow_label',
    'sgid_index',
    'hop_limit',
    'traffic_class',
]
struct_ibv_global_route._fields_ = [
    ('dgid', union_ibv_gid),
    ('flow_label', c_uint32),
    ('sgid_index', c_uint8),
    ('hop_limit', c_uint8),
    ('traffic_class', c_uint8),
]

# verbs.h: 337
class struct_ibv_grh(Structure):
    pass

struct_ibv_grh.__slots__ = [
    'version_tclass_flow',
    'paylen',
    'next_hdr',
    'hop_limit',
    'sgid',
    'dgid',
]
struct_ibv_grh._fields_ = [
    ('version_tclass_flow', c_uint32),
    ('paylen', c_uint16),
    ('next_hdr', c_uint8),
    ('hop_limit', c_uint8),
    ('sgid', union_ibv_gid),
    ('dgid', union_ibv_gid),
]

enum_ibv_rate = c_int # verbs.h: 346

IBV_RATE_MAX = 0 # verbs.h: 346

IBV_RATE_2_5_GBPS = 2 # verbs.h: 346

IBV_RATE_5_GBPS = 5 # verbs.h: 346

IBV_RATE_10_GBPS = 3 # verbs.h: 346

IBV_RATE_20_GBPS = 6 # verbs.h: 346

IBV_RATE_30_GBPS = 4 # verbs.h: 346

IBV_RATE_40_GBPS = 7 # verbs.h: 346

IBV_RATE_60_GBPS = 8 # verbs.h: 346

IBV_RATE_80_GBPS = 9 # verbs.h: 346

IBV_RATE_120_GBPS = 10 # verbs.h: 346

IBV_RATE_14_GBPS = 11 # verbs.h: 346

IBV_RATE_56_GBPS = 12 # verbs.h: 346

IBV_RATE_112_GBPS = 13 # verbs.h: 346

IBV_RATE_168_GBPS = 14 # verbs.h: 346

IBV_RATE_25_GBPS = 15 # verbs.h: 346

IBV_RATE_100_GBPS = 16 # verbs.h: 346

IBV_RATE_200_GBPS = 17 # verbs.h: 346

IBV_RATE_300_GBPS = 18 # verbs.h: 346

# verbs.h: 373
if hasattr(_libs['libibverbs.so.1'], 'ibv_rate_to_mult'):
    ibv_rate_to_mult = _libs['libibverbs.so.1'].ibv_rate_to_mult
    ibv_rate_to_mult.argtypes = [enum_ibv_rate]
    ibv_rate_to_mult.restype = c_int

# verbs.h: 379
if hasattr(_libs['libibverbs.so.1'], 'mult_to_ibv_rate'):
    mult_to_ibv_rate = _libs['libibverbs.so.1'].mult_to_ibv_rate
    mult_to_ibv_rate.argtypes = [c_int]
    mult_to_ibv_rate.restype = enum_ibv_rate

# verbs.h: 386
if hasattr(_libs['libibverbs.so.1'], 'ibv_rate_to_mbps'):
    ibv_rate_to_mbps = _libs['libibverbs.so.1'].ibv_rate_to_mbps
    ibv_rate_to_mbps.argtypes = [enum_ibv_rate]
    ibv_rate_to_mbps.restype = c_int

# verbs.h: 392
if hasattr(_libs['libibverbs.so.1'], 'mbps_to_ibv_rate'):
    mbps_to_ibv_rate = _libs['libibverbs.so.1'].mbps_to_ibv_rate
    mbps_to_ibv_rate.argtypes = [c_int]
    mbps_to_ibv_rate.restype = enum_ibv_rate

# verbs.h: 394
class struct_ibv_ah_attr(Structure):
    pass

struct_ibv_ah_attr.__slots__ = [
    'grh',
    'dlid',
    'sl',
    'src_path_bits',
    'static_rate',
    'is_global',
    'port_num',
]
struct_ibv_ah_attr._fields_ = [
    ('grh', struct_ibv_global_route),
    ('dlid', c_uint16),
    ('sl', c_uint8),
    ('src_path_bits', c_uint8),
    ('static_rate', c_uint8),
    ('is_global', c_uint8),
    ('port_num', c_uint8),
]

enum_ibv_srq_attr_mask = c_int # verbs.h: 404

IBV_SRQ_MAX_WR = (1 << 0) # verbs.h: 404

IBV_SRQ_LIMIT = (1 << 1) # verbs.h: 404

# verbs.h: 409
class struct_ibv_srq_attr(Structure):
    pass

struct_ibv_srq_attr.__slots__ = [
    'max_wr',
    'max_sge',
    'srq_limit',
]
struct_ibv_srq_attr._fields_ = [
    ('max_wr', c_uint32),
    ('max_sge', c_uint32),
    ('srq_limit', c_uint32),
]

# verbs.h: 415
class struct_ibv_srq_init_attr(Structure):
    pass

struct_ibv_srq_init_attr.__slots__ = [
    'srq_context',
    'attr',
]
struct_ibv_srq_init_attr._fields_ = [
    ('srq_context', POINTER(None)),
    ('attr', struct_ibv_srq_attr),
]

enum_ibv_qp_type = c_int # verbs.h: 420

IBV_QPT_RC = 2 # verbs.h: 420

IBV_QPT_UC = (IBV_QPT_RC + 1) # verbs.h: 420

IBV_QPT_UD = (IBV_QPT_UC + 1) # verbs.h: 420

IBV_QPT_RAW_PACKET = 8 # verbs.h: 420

# verbs.h: 427
class struct_ibv_qp_cap(Structure):
    pass

struct_ibv_qp_cap.__slots__ = [
    'max_send_wr',
    'max_recv_wr',
    'max_send_sge',
    'max_recv_sge',
    'max_inline_data',
]
struct_ibv_qp_cap._fields_ = [
    ('max_send_wr', c_uint32),
    ('max_recv_wr', c_uint32),
    ('max_send_sge', c_uint32),
    ('max_recv_sge', c_uint32),
    ('max_inline_data', c_uint32),
]

# verbs.h: 435
class struct_ibv_qp_init_attr(Structure):
    pass

struct_ibv_qp_init_attr.__slots__ = [
    'qp_context',
    'send_cq',
    'recv_cq',
    'srq',
    'cap',
    'qp_type',
    'sq_sig_all',
]
struct_ibv_qp_init_attr._fields_ = [
    ('qp_context', POINTER(None)),
    ('send_cq', POINTER(struct_ibv_cq)),
    ('recv_cq', POINTER(struct_ibv_cq)),
    ('srq', POINTER(struct_ibv_srq)),
    ('cap', struct_ibv_qp_cap),
    ('qp_type', enum_ibv_qp_type),
    ('sq_sig_all', c_int),
]

enum_ibv_qp_attr_mask = c_int # verbs.h: 445

IBV_QP_STATE = (1 << 0) # verbs.h: 445

IBV_QP_CUR_STATE = (1 << 1) # verbs.h: 445

IBV_QP_EN_SQD_ASYNC_NOTIFY = (1 << 2) # verbs.h: 445

IBV_QP_ACCESS_FLAGS = (1 << 3) # verbs.h: 445

IBV_QP_PKEY_INDEX = (1 << 4) # verbs.h: 445

IBV_QP_PORT = (1 << 5) # verbs.h: 445

IBV_QP_QKEY = (1 << 6) # verbs.h: 445

IBV_QP_AV = (1 << 7) # verbs.h: 445

IBV_QP_PATH_MTU = (1 << 8) # verbs.h: 445

IBV_QP_TIMEOUT = (1 << 9) # verbs.h: 445

IBV_QP_RETRY_CNT = (1 << 10) # verbs.h: 445

IBV_QP_RNR_RETRY = (1 << 11) # verbs.h: 445

IBV_QP_RQ_PSN = (1 << 12) # verbs.h: 445

IBV_QP_MAX_QP_RD_ATOMIC = (1 << 13) # verbs.h: 445

IBV_QP_ALT_PATH = (1 << 14) # verbs.h: 445

IBV_QP_MIN_RNR_TIMER = (1 << 15) # verbs.h: 445

IBV_QP_SQ_PSN = (1 << 16) # verbs.h: 445

IBV_QP_MAX_DEST_RD_ATOMIC = (1 << 17) # verbs.h: 445

IBV_QP_PATH_MIG_STATE = (1 << 18) # verbs.h: 445

IBV_QP_CAP = (1 << 19) # verbs.h: 445

IBV_QP_DEST_QPN = (1 << 20) # verbs.h: 445

enum_ibv_qp_state = c_int # verbs.h: 469

IBV_QPS_RESET = 0 # verbs.h: 469

IBV_QPS_INIT = (IBV_QPS_RESET + 1) # verbs.h: 469

IBV_QPS_RTR = (IBV_QPS_INIT + 1) # verbs.h: 469

IBV_QPS_RTS = (IBV_QPS_RTR + 1) # verbs.h: 469

IBV_QPS_SQD = (IBV_QPS_RTS + 1) # verbs.h: 469

IBV_QPS_SQE = (IBV_QPS_SQD + 1) # verbs.h: 469

IBV_QPS_ERR = (IBV_QPS_SQE + 1) # verbs.h: 469

enum_ibv_mig_state = c_int # verbs.h: 479

IBV_MIG_MIGRATED = 0 # verbs.h: 479

IBV_MIG_REARM = (IBV_MIG_MIGRATED + 1) # verbs.h: 479

IBV_MIG_ARMED = (IBV_MIG_REARM + 1) # verbs.h: 479

# verbs.h: 485
class struct_ibv_qp_attr(Structure):
    pass

struct_ibv_qp_attr.__slots__ = [
    'qp_state',
    'cur_qp_state',
    'path_mtu',
    'path_mig_state',
    'qkey',
    'rq_psn',
    'sq_psn',
    'dest_qp_num',
    'qp_access_flags',
    'cap',
    'ah_attr',
    'alt_ah_attr',
    'pkey_index',
    'alt_pkey_index',
    'en_sqd_async_notify',
    'sq_draining',
    'max_rd_atomic',
    'max_dest_rd_atomic',
    'min_rnr_timer',
    'port_num',
    'timeout',
    'retry_cnt',
    'rnr_retry',
    'alt_port_num',
    'alt_timeout',
]
struct_ibv_qp_attr._fields_ = [
    ('qp_state', enum_ibv_qp_state),
    ('cur_qp_state', enum_ibv_qp_state),
    ('path_mtu', enum_ibv_mtu),
    ('path_mig_state', enum_ibv_mig_state),
    ('qkey', c_uint32),
    ('rq_psn', c_uint32),
    ('sq_psn', c_uint32),
    ('dest_qp_num', c_uint32),
    ('qp_access_flags', c_int),
    ('cap', struct_ibv_qp_cap),
    ('ah_attr', struct_ibv_ah_attr),
    ('alt_ah_attr', struct_ibv_ah_attr),
    ('pkey_index', c_uint16),
    ('alt_pkey_index', c_uint16),
    ('en_sqd_async_notify', c_uint8),
    ('sq_draining', c_uint8),
    ('max_rd_atomic', c_uint8),
    ('max_dest_rd_atomic', c_uint8),
    ('min_rnr_timer', c_uint8),
    ('port_num', c_uint8),
    ('timeout', c_uint8),
    ('retry_cnt', c_uint8),
    ('rnr_retry', c_uint8),
    ('alt_port_num', c_uint8),
    ('alt_timeout', c_uint8),
]

enum_ibv_wr_opcode = c_int # verbs.h: 513

IBV_WR_RDMA_WRITE = 0 # verbs.h: 513

IBV_WR_RDMA_WRITE_WITH_IMM = (IBV_WR_RDMA_WRITE + 1) # verbs.h: 513

IBV_WR_SEND = (IBV_WR_RDMA_WRITE_WITH_IMM + 1) # verbs.h: 513

IBV_WR_SEND_WITH_IMM = (IBV_WR_SEND + 1) # verbs.h: 513

IBV_WR_RDMA_READ = (IBV_WR_SEND_WITH_IMM + 1) # verbs.h: 513

IBV_WR_ATOMIC_CMP_AND_SWP = (IBV_WR_RDMA_READ + 1) # verbs.h: 513

IBV_WR_ATOMIC_FETCH_AND_ADD = (IBV_WR_ATOMIC_CMP_AND_SWP + 1) # verbs.h: 513

enum_ibv_send_flags = c_int # verbs.h: 523

IBV_SEND_FENCE = (1 << 0) # verbs.h: 523

IBV_SEND_SIGNALED = (1 << 1) # verbs.h: 523

IBV_SEND_SOLICITED = (1 << 2) # verbs.h: 523

IBV_SEND_INLINE = (1 << 3) # verbs.h: 523

# verbs.h: 530
class struct_ibv_sge(Structure):
    pass

struct_ibv_sge.__slots__ = [
    'addr',
    'length',
    'lkey',
]
struct_ibv_sge._fields_ = [
    ('addr', c_uint64),
    ('length', c_uint32),
    ('lkey', c_uint32),
]

# verbs.h: 536
class struct_ibv_send_wr(Structure):
    pass

# verbs.h: 545
class struct_anon_28(Structure):
    pass

struct_anon_28.__slots__ = [
    'remote_addr',
    'rkey',
]
struct_anon_28._fields_ = [
    ('remote_addr', c_uint64),
    ('rkey', c_uint32),
]

# verbs.h: 549
class struct_anon_29(Structure):
    pass

struct_anon_29.__slots__ = [
    'remote_addr',
    'compare_add',
    'swap',
    'rkey',
]
struct_anon_29._fields_ = [
    ('remote_addr', c_uint64),
    ('compare_add', c_uint64),
    ('swap', c_uint64),
    ('rkey', c_uint32),
]

# verbs.h: 626
class struct_ibv_ah(Structure):
    pass

# verbs.h: 555
class struct_anon_30(Structure):
    pass

struct_anon_30.__slots__ = [
    'ah',
    'remote_qpn',
    'remote_qkey',
]
struct_anon_30._fields_ = [
    ('ah', POINTER(struct_ibv_ah)),
    ('remote_qpn', c_uint32),
    ('remote_qkey', c_uint32),
]

# verbs.h: 544
class union_anon_31(Union):
    pass

union_anon_31.__slots__ = [
    'rdma',
    'atomic',
    'ud',
]
union_anon_31._fields_ = [
    ('rdma', struct_anon_28),
    ('atomic', struct_anon_29),
    ('ud', struct_anon_30),
]

struct_ibv_send_wr.__slots__ = [
    'wr_id',
    'next',
    'sg_list',
    'num_sge',
    'opcode',
    'send_flags',
    'imm_data',
    'wr',
]
struct_ibv_send_wr._fields_ = [
    ('wr_id', c_uint64),
    ('next', POINTER(struct_ibv_send_wr)),
    ('sg_list', POINTER(struct_ibv_sge)),
    ('num_sge', c_int),
    ('opcode', enum_ibv_wr_opcode),
    ('send_flags', c_int),
    ('imm_data', c_uint32),
    ('wr', union_anon_31),
]

# verbs.h: 563
class struct_ibv_recv_wr(Structure):
    pass

struct_ibv_recv_wr.__slots__ = [
    'wr_id',
    'next',
    'sg_list',
    'num_sge',
]
struct_ibv_recv_wr._fields_ = [
    ('wr_id', c_uint64),
    ('next', POINTER(struct_ibv_recv_wr)),
    ('sg_list', POINTER(struct_ibv_sge)),
    ('num_sge', c_int),
]

# verbs.h: 570
class struct_ibv_mw_bind(Structure):
    pass

struct_ibv_mw_bind.__slots__ = [
    'wr_id',
    'mr',
    'addr',
    'length',
    'send_flags',
    'mw_access_flags',
]
struct_ibv_mw_bind._fields_ = [
    ('wr_id', c_uint64),
    ('mr', POINTER(struct_ibv_mr)),
    ('addr', POINTER(None)),
    ('length', c_size_t),
    ('send_flags', c_int),
    ('mw_access_flags', c_int),
]

struct_ibv_srq.__slots__ = [
    'context',
    'srq_context',
    'pd',
    'handle',
    'mutex',
    'cond',
    'events_completed',
]
struct_ibv_srq._fields_ = [
    ('context', POINTER(struct_ibv_context)),
    ('srq_context', POINTER(None)),
    ('pd', POINTER(struct_ibv_pd)),
    ('handle', c_uint32),
    ('mutex', pthread_mutex_t),
    ('cond', pthread_cond_t),
    ('events_completed', c_uint32),
]

struct_ibv_qp.__slots__ = [
    'context',
    'qp_context',
    'pd',
    'send_cq',
    'recv_cq',
    'srq',
    'handle',
    'qp_num',
    'state',
    'qp_type',
    'mutex',
    'cond',
    'events_completed',
]
struct_ibv_qp._fields_ = [
    ('context', POINTER(struct_ibv_context)),
    ('qp_context', POINTER(None)),
    ('pd', POINTER(struct_ibv_pd)),
    ('send_cq', POINTER(struct_ibv_cq)),
    ('recv_cq', POINTER(struct_ibv_cq)),
    ('srq', POINTER(struct_ibv_srq)),
    ('handle', c_uint32),
    ('qp_num', c_uint32),
    ('state', enum_ibv_qp_state),
    ('qp_type', enum_ibv_qp_type),
    ('mutex', pthread_mutex_t),
    ('cond', pthread_cond_t),
    ('events_completed', c_uint32),
]

# verbs.h: 607
class struct_ibv_comp_channel(Structure):
    pass

struct_ibv_comp_channel.__slots__ = [
    'context',
    'fd',
    'refcnt',
]
struct_ibv_comp_channel._fields_ = [
    ('context', POINTER(struct_ibv_context)),
    ('fd', c_int),
    ('refcnt', c_int),
]

struct_ibv_cq.__slots__ = [
    'context',
    'channel',
    'cq_context',
    'handle',
    'cqe',
    'mutex',
    'cond',
    'comp_events_completed',
    'async_events_completed',
]
struct_ibv_cq._fields_ = [
    ('context', POINTER(struct_ibv_context)),
    ('channel', POINTER(struct_ibv_comp_channel)),
    ('cq_context', POINTER(None)),
    ('handle', c_uint32),
    ('cqe', c_int),
    ('mutex', pthread_mutex_t),
    ('cond', pthread_cond_t),
    ('comp_events_completed', c_uint32),
    ('async_events_completed', c_uint32),
]

struct_ibv_ah.__slots__ = [
    'context',
    'pd',
    'handle',
]
struct_ibv_ah._fields_ = [
    ('context', POINTER(struct_ibv_context)),
    ('pd', POINTER(struct_ibv_pd)),
    ('handle', c_uint32),
]

# verbs.h: 645
class struct_ibv_device(Structure):
    pass

# verbs.h: 635
class struct_ibv_device_ops(Structure):
    pass

struct_ibv_device_ops.__slots__ = [
    'alloc_context',
    'free_context',
]
struct_ibv_device_ops._fields_ = [
    ('alloc_context', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_context)), POINTER(struct_ibv_device), c_int)),
    ('free_context', CFUNCTYPE(UNCHECKED(None), POINTER(struct_ibv_context))),
]

enum_anon_32 = c_int # verbs.h: 640

IBV_SYSFS_NAME_MAX = 64 # verbs.h: 640

IBV_SYSFS_PATH_MAX = 256 # verbs.h: 640

struct_ibv_device.__slots__ = [
    'ops',
    'node_type',
    'transport_type',
    'name',
    'dev_name',
    'dev_path',
    'ibdev_path',
]
struct_ibv_device._fields_ = [
    ('ops', struct_ibv_device_ops),
    ('node_type', enum_ibv_node_type),
    ('transport_type', enum_ibv_transport_type),
    ('name', c_char * IBV_SYSFS_NAME_MAX),
    ('dev_name', c_char * IBV_SYSFS_NAME_MAX),
    ('dev_path', c_char * IBV_SYSFS_PATH_MAX),
    ('ibdev_path', c_char * IBV_SYSFS_PATH_MAX),
]

# verbs.h: 659
class struct_ibv_context_ops(Structure):
    pass

struct_ibv_context_ops.__slots__ = [
    'query_device',
    'query_port',
    'alloc_pd',
    'dealloc_pd',
    'reg_mr',
    'rereg_mr',
    'dereg_mr',
    'alloc_mw',
    'bind_mw',
    'dealloc_mw',
    'create_cq',
    'poll_cq',
    'req_notify_cq',
    'cq_event',
    'resize_cq',
    'destroy_cq',
    'create_srq',
    'modify_srq',
    'query_srq',
    'destroy_srq',
    'post_srq_recv',
    'create_qp',
    'query_qp',
    'modify_qp',
    'destroy_qp',
    'post_send',
    'post_recv',
    'create_ah',
    'destroy_ah',
    'attach_mcast',
    'detach_mcast',
    'async_event',
]
struct_ibv_context_ops._fields_ = [
    ('query_device', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_context), POINTER(struct_ibv_device_attr))),
    ('query_port', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_context), c_uint8, POINTER(struct_ibv_port_attr))),
    ('alloc_pd', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_pd)), POINTER(struct_ibv_context))),
    ('dealloc_pd', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_pd))),
    ('reg_mr', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_mr)), POINTER(struct_ibv_pd), POINTER(None), c_size_t, c_int)),
    ('rereg_mr', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_mr)), POINTER(struct_ibv_mr), c_int, POINTER(struct_ibv_pd), POINTER(None), c_size_t, c_int)),
    ('dereg_mr', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_mr))),
    ('alloc_mw', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_mw)), POINTER(struct_ibv_pd), enum_ibv_mw_type)),
    ('bind_mw', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_qp), POINTER(struct_ibv_mw), POINTER(struct_ibv_mw_bind))),
    ('dealloc_mw', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_mw))),
    ('create_cq', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_cq)), POINTER(struct_ibv_context), c_int, POINTER(struct_ibv_comp_channel), c_int)),
    ('poll_cq', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_cq), c_int, POINTER(struct_ibv_wc))),
    ('req_notify_cq', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_cq), c_int)),
    ('cq_event', CFUNCTYPE(UNCHECKED(None), POINTER(struct_ibv_cq))),
    ('resize_cq', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_cq), c_int)),
    ('destroy_cq', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_cq))),
    ('create_srq', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_srq)), POINTER(struct_ibv_pd), POINTER(struct_ibv_srq_init_attr))),
    ('modify_srq', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_srq), POINTER(struct_ibv_srq_attr), c_int)),
    ('query_srq', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_srq), POINTER(struct_ibv_srq_attr))),
    ('destroy_srq', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_srq))),
    ('post_srq_recv', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_srq), POINTER(struct_ibv_recv_wr), POINTER(POINTER(struct_ibv_recv_wr)))),
    ('create_qp', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_qp)), POINTER(struct_ibv_pd), POINTER(struct_ibv_qp_init_attr))),
    ('query_qp', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_qp), POINTER(struct_ibv_qp_attr), c_int, POINTER(struct_ibv_qp_init_attr))),
    ('modify_qp', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_qp), POINTER(struct_ibv_qp_attr), c_int)),
    ('destroy_qp', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_qp))),
    ('post_send', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_qp), POINTER(struct_ibv_send_wr), POINTER(POINTER(struct_ibv_send_wr)))),
    ('post_recv', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_qp), POINTER(struct_ibv_recv_wr), POINTER(POINTER(struct_ibv_recv_wr)))),
    ('create_ah', CFUNCTYPE(UNCHECKED(POINTER(struct_ibv_ah)), POINTER(struct_ibv_pd), POINTER(struct_ibv_ah_attr))),
    ('destroy_ah', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_ah))),
    ('attach_mcast', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_qp), POINTER(union_ibv_gid), c_uint16)),
    ('detach_mcast', CFUNCTYPE(UNCHECKED(c_int), POINTER(struct_ibv_qp), POINTER(union_ibv_gid), c_uint16)),
    ('async_event', CFUNCTYPE(UNCHECKED(None), POINTER(struct_ibv_async_event))),
]

struct_ibv_context.__slots__ = [
    'device',
    'ops',
    'cmd_fd',
    'async_fd',
    'num_comp_vectors',
    'mutex',
    'abi_compat',
]
struct_ibv_context._fields_ = [
    ('device', POINTER(struct_ibv_device)),
    ('ops', struct_ibv_context_ops),
    ('cmd_fd', c_int),
    ('async_fd', c_int),
    ('num_comp_vectors', c_int),
    ('mutex', pthread_mutex_t),
    ('abi_compat', POINTER(None)),
]

# verbs.h: 735
if hasattr(_libs['libibverbs.so.1'], 'ibv_get_device_list'):
    ibv_get_device_list = _libs['libibverbs.so.1'].ibv_get_device_list
    ibv_get_device_list.argtypes = [POINTER(c_int)]
    ibv_get_device_list.restype = POINTER(POINTER(struct_ibv_device))

# verbs.h: 745
if hasattr(_libs['libibverbs.so.1'], 'ibv_free_device_list'):
    ibv_free_device_list = _libs['libibverbs.so.1'].ibv_free_device_list
    ibv_free_device_list.argtypes = [POINTER(POINTER(struct_ibv_device))]
    ibv_free_device_list.restype = None

# verbs.h: 750
if hasattr(_libs['libibverbs.so.1'], 'ibv_get_device_name'):
    ibv_get_device_name = _libs['libibverbs.so.1'].ibv_get_device_name
    ibv_get_device_name.argtypes = [POINTER(struct_ibv_device)]
    if sizeof(c_int) == sizeof(c_void_p):
        ibv_get_device_name.restype = ReturnString
    else:
        ibv_get_device_name.restype = String
        ibv_get_device_name.errcheck = ReturnString

# verbs.h: 755
if hasattr(_libs['libibverbs.so.1'], 'ibv_get_device_guid'):
    ibv_get_device_guid = _libs['libibverbs.so.1'].ibv_get_device_guid
    ibv_get_device_guid.argtypes = [POINTER(struct_ibv_device)]
    ibv_get_device_guid.restype = c_uint64

# verbs.h: 760
if hasattr(_libs['libibverbs.so.1'], 'ibv_open_device'):
    ibv_open_device = _libs['libibverbs.so.1'].ibv_open_device
    ibv_open_device.argtypes = [POINTER(struct_ibv_device)]
    ibv_open_device.restype = POINTER(struct_ibv_context)

# verbs.h: 765
if hasattr(_libs['libibverbs.so.1'], 'ibv_close_device'):
    ibv_close_device = _libs['libibverbs.so.1'].ibv_close_device
    ibv_close_device.argtypes = [POINTER(struct_ibv_context)]
    ibv_close_device.restype = c_int

# verbs.h: 774
if hasattr(_libs['libibverbs.so.1'], 'ibv_get_async_event'):
    ibv_get_async_event = _libs['libibverbs.so.1'].ibv_get_async_event
    ibv_get_async_event.argtypes = [POINTER(struct_ibv_context), POINTER(struct_ibv_async_event)]
    ibv_get_async_event.restype = c_int

# verbs.h: 787
if hasattr(_libs['libibverbs.so.1'], 'ibv_ack_async_event'):
    ibv_ack_async_event = _libs['libibverbs.so.1'].ibv_ack_async_event
    ibv_ack_async_event.argtypes = [POINTER(struct_ibv_async_event)]
    ibv_ack_async_event.restype = None

# verbs.h: 792
if hasattr(_libs['libibverbs.so.1'], 'ibv_query_device'):
    ibv_query_device = _libs['libibverbs.so.1'].ibv_query_device
    ibv_query_device.argtypes = [POINTER(struct_ibv_context), POINTER(struct_ibv_device_attr)]
    ibv_query_device.restype = c_int

# verbs.h: 798
if hasattr(_libs['libibverbs.so.1'], 'ibv_query_port'):
    ibv_query_port = _libs['libibverbs.so.1'].ibv_query_port
    ibv_query_port.argtypes = [POINTER(struct_ibv_context), c_uint8, POINTER(struct_ibv_port_attr)]
    ibv_query_port.restype = c_int

# verbs.h: 818
if hasattr(_libs['libibverbs.so.1'], 'ibv_query_gid'):
    ibv_query_gid = _libs['libibverbs.so.1'].ibv_query_gid
    ibv_query_gid.argtypes = [POINTER(struct_ibv_context), c_uint8, c_int, POINTER(union_ibv_gid)]
    ibv_query_gid.restype = c_int

# verbs.h: 824
if hasattr(_libs['libibverbs.so.1'], 'ibv_query_pkey'):
    ibv_query_pkey = _libs['libibverbs.so.1'].ibv_query_pkey
    ibv_query_pkey.argtypes = [POINTER(struct_ibv_context), c_uint8, c_int, POINTER(c_uint16)]
    ibv_query_pkey.restype = c_int

# verbs.h: 830
if hasattr(_libs['libibverbs.so.1'], 'ibv_alloc_pd'):
    ibv_alloc_pd = _libs['libibverbs.so.1'].ibv_alloc_pd
    ibv_alloc_pd.argtypes = [POINTER(struct_ibv_context)]
    ibv_alloc_pd.restype = POINTER(struct_ibv_pd)

# verbs.h: 835
if hasattr(_libs['libibverbs.so.1'], 'ibv_dealloc_pd'):
    ibv_dealloc_pd = _libs['libibverbs.so.1'].ibv_dealloc_pd
    ibv_dealloc_pd.argtypes = [POINTER(struct_ibv_pd)]
    ibv_dealloc_pd.restype = c_int

# verbs.h: 840
if hasattr(_libs['libibverbs.so.1'], 'ibv_reg_mr'):
    ibv_reg_mr = _libs['libibverbs.so.1'].ibv_reg_mr
    ibv_reg_mr.argtypes = [POINTER(struct_ibv_pd), POINTER(None), c_size_t, c_int]
    ibv_reg_mr.restype = POINTER(struct_ibv_mr)

# verbs.h: 846
if hasattr(_libs['libibverbs.so.1'], 'ibv_dereg_mr'):
    ibv_dereg_mr = _libs['libibverbs.so.1'].ibv_dereg_mr
    ibv_dereg_mr.argtypes = [POINTER(struct_ibv_mr)]
    ibv_dereg_mr.restype = c_int

# verbs.h: 851
if hasattr(_libs['libibverbs.so.1'], 'ibv_create_comp_channel'):
    ibv_create_comp_channel = _libs['libibverbs.so.1'].ibv_create_comp_channel
    ibv_create_comp_channel.argtypes = [POINTER(struct_ibv_context)]
    ibv_create_comp_channel.restype = POINTER(struct_ibv_comp_channel)

# verbs.h: 856
if hasattr(_libs['libibverbs.so.1'], 'ibv_destroy_comp_channel'):
    ibv_destroy_comp_channel = _libs['libibverbs.so.1'].ibv_destroy_comp_channel
    ibv_destroy_comp_channel.argtypes = [POINTER(struct_ibv_comp_channel)]
    ibv_destroy_comp_channel.restype = c_int

# verbs.h: 868
if hasattr(_libs['libibverbs.so.1'], 'ibv_create_cq'):
    ibv_create_cq = _libs['libibverbs.so.1'].ibv_create_cq
    ibv_create_cq.argtypes = [POINTER(struct_ibv_context), c_int, POINTER(None), POINTER(struct_ibv_comp_channel), c_int]
    ibv_create_cq.restype = POINTER(struct_ibv_cq)

# verbs.h: 880
if hasattr(_libs['libibverbs.so.1'], 'ibv_resize_cq'):
    ibv_resize_cq = _libs['libibverbs.so.1'].ibv_resize_cq
    ibv_resize_cq.argtypes = [POINTER(struct_ibv_cq), c_int]
    ibv_resize_cq.restype = c_int

# verbs.h: 885
if hasattr(_libs['libibverbs.so.1'], 'ibv_destroy_cq'):
    ibv_destroy_cq = _libs['libibverbs.so.1'].ibv_destroy_cq
    ibv_destroy_cq.argtypes = [POINTER(struct_ibv_cq)]
    ibv_destroy_cq.restype = c_int

# verbs.h: 896
if hasattr(_libs['libibverbs.so.1'], 'ibv_get_cq_event'):
    ibv_get_cq_event = _libs['libibverbs.so.1'].ibv_get_cq_event
    ibv_get_cq_event.argtypes = [POINTER(struct_ibv_comp_channel), POINTER(POINTER(struct_ibv_cq)), POINTER(POINTER(None))]
    ibv_get_cq_event.restype = c_int

# verbs.h: 912
if hasattr(_libs['libibverbs.so.1'], 'ibv_ack_cq_events'):
    ibv_ack_cq_events = _libs['libibverbs.so.1'].ibv_ack_cq_events
    ibv_ack_cq_events.argtypes = [POINTER(struct_ibv_cq), c_uint]
    ibv_ack_cq_events.restype = None

# verbs.h: 957
if hasattr(_libs['libibverbs.so.1'], 'ibv_create_srq'):
    ibv_create_srq = _libs['libibverbs.so.1'].ibv_create_srq
    ibv_create_srq.argtypes = [POINTER(struct_ibv_pd), POINTER(struct_ibv_srq_init_attr)]
    ibv_create_srq.restype = POINTER(struct_ibv_srq)

# verbs.h: 972
if hasattr(_libs['libibverbs.so.1'], 'ibv_modify_srq'):
    ibv_modify_srq = _libs['libibverbs.so.1'].ibv_modify_srq
    ibv_modify_srq.argtypes = [POINTER(struct_ibv_srq), POINTER(struct_ibv_srq_attr), c_int]
    ibv_modify_srq.restype = c_int

# verbs.h: 982
if hasattr(_libs['libibverbs.so.1'], 'ibv_query_srq'):
    ibv_query_srq = _libs['libibverbs.so.1'].ibv_query_srq
    ibv_query_srq.argtypes = [POINTER(struct_ibv_srq), POINTER(struct_ibv_srq_attr)]
    ibv_query_srq.restype = c_int

# verbs.h: 988
if hasattr(_libs['libibverbs.so.1'], 'ibv_destroy_srq'):
    ibv_destroy_srq = _libs['libibverbs.so.1'].ibv_destroy_srq
    ibv_destroy_srq.argtypes = [POINTER(struct_ibv_srq)]
    ibv_destroy_srq.restype = c_int

# verbs.h: 1007
if hasattr(_libs['libibverbs.so.1'], 'ibv_create_qp'):
    ibv_create_qp = _libs['libibverbs.so.1'].ibv_create_qp
    ibv_create_qp.argtypes = [POINTER(struct_ibv_pd), POINTER(struct_ibv_qp_init_attr)]
    ibv_create_qp.restype = POINTER(struct_ibv_qp)

# verbs.h: 1013
if hasattr(_libs['libibverbs.so.1'], 'ibv_modify_qp'):
    ibv_modify_qp = _libs['libibverbs.so.1'].ibv_modify_qp
    ibv_modify_qp.argtypes = [POINTER(struct_ibv_qp), POINTER(struct_ibv_qp_attr), c_int]
    ibv_modify_qp.restype = c_int

# verbs.h: 1027
if hasattr(_libs['libibverbs.so.1'], 'ibv_query_qp'):
    ibv_query_qp = _libs['libibverbs.so.1'].ibv_query_qp
    ibv_query_qp.argtypes = [POINTER(struct_ibv_qp), POINTER(struct_ibv_qp_attr), c_int, POINTER(struct_ibv_qp_init_attr)]
    ibv_query_qp.restype = c_int

# verbs.h: 1034
if hasattr(_libs['libibverbs.so.1'], 'ibv_destroy_qp'):
    ibv_destroy_qp = _libs['libibverbs.so.1'].ibv_destroy_qp
    ibv_destroy_qp.argtypes = [POINTER(struct_ibv_qp)]
    ibv_destroy_qp.restype = c_int

# verbs.h: 1060
if hasattr(_libs['libibverbs.so.1'], 'ibv_create_ah'):
    ibv_create_ah = _libs['libibverbs.so.1'].ibv_create_ah
    ibv_create_ah.argtypes = [POINTER(struct_ibv_pd), POINTER(struct_ibv_ah_attr)]
    ibv_create_ah.restype = POINTER(struct_ibv_ah)

# verbs.h: 1073
if hasattr(_libs['libibverbs.so.1'], 'ibv_init_ah_from_wc'):
    ibv_init_ah_from_wc = _libs['libibverbs.so.1'].ibv_init_ah_from_wc
    ibv_init_ah_from_wc.argtypes = [POINTER(struct_ibv_context), c_uint8, POINTER(struct_ibv_wc), POINTER(struct_ibv_grh), POINTER(struct_ibv_ah_attr)]
    ibv_init_ah_from_wc.restype = c_int

# verbs.h: 1089
if hasattr(_libs['libibverbs.so.1'], 'ibv_create_ah_from_wc'):
    ibv_create_ah_from_wc = _libs['libibverbs.so.1'].ibv_create_ah_from_wc
    ibv_create_ah_from_wc.argtypes = [POINTER(struct_ibv_pd), POINTER(struct_ibv_wc), POINTER(struct_ibv_grh), c_uint8]
    ibv_create_ah_from_wc.restype = POINTER(struct_ibv_ah)

# verbs.h: 1095
if hasattr(_libs['libibverbs.so.1'], 'ibv_destroy_ah'):
    ibv_destroy_ah = _libs['libibverbs.so.1'].ibv_destroy_ah
    ibv_destroy_ah.argtypes = [POINTER(struct_ibv_ah)]
    ibv_destroy_ah.restype = c_int

# verbs.h: 1108
if hasattr(_libs['libibverbs.so.1'], 'ibv_attach_mcast'):
    ibv_attach_mcast = _libs['libibverbs.so.1'].ibv_attach_mcast
    ibv_attach_mcast.argtypes = [POINTER(struct_ibv_qp), POINTER(union_ibv_gid), c_uint16]
    ibv_attach_mcast.restype = c_int

# verbs.h: 1116
if hasattr(_libs['libibverbs.so.1'], 'ibv_detach_mcast'):
    ibv_detach_mcast = _libs['libibverbs.so.1'].ibv_detach_mcast
    ibv_detach_mcast.argtypes = [POINTER(struct_ibv_qp), POINTER(union_ibv_gid), c_uint16]
    ibv_detach_mcast.restype = c_int

# verbs.h: 1124
if hasattr(_libs['libibverbs.so.1'], 'ibv_fork_init'):
    ibv_fork_init = _libs['libibverbs.so.1'].ibv_fork_init
    ibv_fork_init.argtypes = []
    ibv_fork_init.restype = c_int

# verbs.h: 1129
if hasattr(_libs['libibverbs.so.1'], 'ibv_node_type_str'):
    ibv_node_type_str = _libs['libibverbs.so.1'].ibv_node_type_str
    ibv_node_type_str.argtypes = [enum_ibv_node_type]
    if sizeof(c_int) == sizeof(c_void_p):
        ibv_node_type_str.restype = ReturnString
    else:
        ibv_node_type_str.restype = String
        ibv_node_type_str.errcheck = ReturnString

# verbs.h: 1134
if hasattr(_libs['libibverbs.so.1'], 'ibv_port_state_str'):
    ibv_port_state_str = _libs['libibverbs.so.1'].ibv_port_state_str
    ibv_port_state_str.argtypes = [enum_ibv_port_state]
    if sizeof(c_int) == sizeof(c_void_p):
        ibv_port_state_str.restype = ReturnString
    else:
        ibv_port_state_str.restype = String
        ibv_port_state_str.errcheck = ReturnString

# verbs.h: 1139
if hasattr(_libs['libibverbs.so.1'], 'ibv_event_type_str'):
    ibv_event_type_str = _libs['libibverbs.so.1'].ibv_event_type_str
    ibv_event_type_str.argtypes = [enum_ibv_event_type]
    if sizeof(c_int) == sizeof(c_void_p):
        ibv_event_type_str.restype = ReturnString
    else:
        ibv_event_type_str.restype = String
        ibv_event_type_str.errcheck = ReturnString

ibv_gid = union_ibv_gid # verbs.h: 58

ibv_device_attr = struct_ibv_device_attr # verbs.h: 104

ibv_port_attr = struct_ibv_port_attr # verbs.h: 170

ibv_cq = struct_ibv_cq # verbs.h: 613

ibv_qp = struct_ibv_qp # verbs.h: 590

ibv_srq = struct_ibv_srq # verbs.h: 579

ibv_async_event = struct_ibv_async_event # verbs.h: 216

ibv_wc = struct_ibv_wc # verbs.h: 272

ibv_context = struct_ibv_context # verbs.h: 717

ibv_pd = struct_ibv_pd # verbs.h: 296

ibv_mr = struct_ibv_mr # verbs.h: 308

ibv_mw = struct_ibv_mw # verbs.h: 323

ibv_global_route = struct_ibv_global_route # verbs.h: 329

ibv_grh = struct_ibv_grh # verbs.h: 337

ibv_ah_attr = struct_ibv_ah_attr # verbs.h: 394

ibv_srq_attr = struct_ibv_srq_attr # verbs.h: 409

ibv_srq_init_attr = struct_ibv_srq_init_attr # verbs.h: 415

ibv_qp_cap = struct_ibv_qp_cap # verbs.h: 427

ibv_qp_init_attr = struct_ibv_qp_init_attr # verbs.h: 435

ibv_qp_attr = struct_ibv_qp_attr # verbs.h: 485

ibv_sge = struct_ibv_sge # verbs.h: 530

ibv_send_wr = struct_ibv_send_wr # verbs.h: 536

ibv_ah = struct_ibv_ah # verbs.h: 626

ibv_recv_wr = struct_ibv_recv_wr # verbs.h: 563

ibv_mw_bind = struct_ibv_mw_bind # verbs.h: 570

ibv_comp_channel = struct_ibv_comp_channel # verbs.h: 607

ibv_device = struct_ibv_device # verbs.h: 645

ibv_device_ops = struct_ibv_device_ops # verbs.h: 635

ibv_context_ops = struct_ibv_context_ops # verbs.h: 659

# No inserted files

