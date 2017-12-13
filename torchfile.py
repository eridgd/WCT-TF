"""
Copyright (c) 2016, Brendan Shillingford
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Modified version of torchfile that is py3 and Windows compatible
Fixes provided by @xdaimon https://github.com/bshillingford/python-torchfile/pull/13
----------------------------------------------------------------------
Mostly direct port of the Lua and C serialization implementation to
Python, depending only on `struct`, `array`, and numpy.

Supported types:
 * `nil` to Python `None`
 * numbers to Python floats, or by default a heuristic changes them to ints or
   longs if they are integral
 * booleans
 * strings: read as byte strings (Python 3) or normal strings (Python 2), like
   lua strings which don't support unicode, and that can contain null chars
 * tables converted to a special dict (*); if they are list-like (i.e. have
   numeric keys from 1 through n) they become a python list by default
 * Torch classes: supports Tensors and Storages, and most classes such as
   modules. Trivially extensible much like the Torch serialization code.
   Trivial torch classes like most `nn.Module` subclasses become
   `TorchObject`s. The `type_handlers` dict contains the mapping from class
   names to reading functions.
 * functions: loaded into the `LuaFunction` `namedtuple`,
   which simply wraps the raw serialized data, i.e. upvalues and code.
   These are mostly useless, but exist so you can deserialize anything.

(*) Since Lua allows you to index a table with a table but Python does not, we
    replace dicts with a subclass that is hashable, and change its
    equality comparison behaviour to compare by reference.
    See `hashable_uniq_dict`.

Currently, the implementation assumes the system-dependent binary Torch
format, but minor refactoring can give support for the ascii format as well.
"""
import struct
from array import array
import numpy as np
import sys
from collections import namedtuple


TYPE_NIL = 0
TYPE_NUMBER = 1
TYPE_STRING = 2
TYPE_TABLE = 3
TYPE_TORCH = 4
TYPE_BOOLEAN = 5
TYPE_FUNCTION = 6
TYPE_RECUR_FUNCTION = 8
LEGACY_TYPE_RECUR_FUNCTION = 7

LuaFunction = namedtuple('LuaFunction',
                         ['size', 'dumped', 'upvalues'])

class mycontainer():
    def __init__(self, val):
        self.val = val
    def __hash__(self):
        return id(self.val)
    def __eq__(self, other):
        return id(self.val) == id(other.val)
    def __ne__(self, other):
        return id(self.val) != id(other.val)

class hashable_uniq_dict(dict):
    """
    Subclass of dict with equality and hashing semantics changed:
    equality and hashing is purely by reference/instance, to match
    the behaviour of lua tables.

    Supports lua-style dot indexing.

    This way, dicts can be keys of other dicts.
    """

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, k):
        for _k,v in self.items():
            if str(_k) == str(k):
                return v

    def __setitem__(self, k, v):
        dict.__setitem__(self, mycontainer(k), v)

    def items(self):
        return [(k.val, v) for k,v in dict.items(self)]

    def keys(self):
        return [k.val for k in dict.keys(self)]

    def values(self):
        return [v for v in dict.values(self)]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def _disabled_binop(self, other):
        raise TypeError(
            'hashable_uniq_dict does not support these comparisons')
    __cmp__ = __ne__ = __le__ = __gt__ = __lt__ = _disabled_binop

class TorchObject(object):
    """
    Simple torch object, used by `add_trivial_class_reader`.
    Supports both forms of lua-style indexing, i.e. getattr and getitem.
    Use the `torch_typename` method to get the object's torch class name.

    Equality is by reference, as usual for lua (and the default for Python
    objects).
    """

    def __init__(self, typename, obj=None, version_number=0):
        self._typename = typename
        self._obj = obj
        self._version_number = version_number

    def __getattr__(self, k):
        if k in self._obj.keys():
            return self._obj[k]
        if isinstance(k, (str, bytes)):
            return self._obj[k.encode('utf8')]
    
    def __getitem__(self, k):
        if k in self._obj.keys():
            return self._obj[k]
        if isinstance(k, (str, bytes)):
            return self._obj[k.encode('utf8')]

    def torch_typename(self):
        return self._typename

    def __repr__(self):
        return "TorchObject(%s, %s)" % (self._typename, repr(self._obj))

    def __str__(self):
        return repr(self)

    def __dir__(self):
        keys = self._obj.keys()
        keys.append('torch_typename')
        return keys


type_handlers = {}


def register_handler(typename):
    def do_register(handler):
        type_handlers[typename] = handler
    return do_register


def add_tensor_reader(typename, dtype):
    def read_tensor_generic(reader, version):
        # https://github.com/torch/torch7/blob/1e86025/generic/Tensor.c#L1249
        ndim = reader.read_int()

        size = reader.read_long_array(ndim)
        stride = reader.read_long_array(ndim)
        storage_offset = reader.read_long() - 1  # 0-indexing
        # read storage:
        storage = reader.read_obj()

        if storage is None or ndim == 0 or len(size) == 0 or len(stride) == 0:
            # empty torch tensor
            return np.empty((0), dtype=dtype)

        # convert stride to numpy style (i.e. in bytes)
        stride = [storage.dtype.itemsize * x for x in stride]

        # create numpy array that indexes into the storage:
        return np.lib.stride_tricks.as_strided(
            storage[storage_offset:],
            shape=size,
            strides=stride)
    type_handlers[typename] = read_tensor_generic
add_tensor_reader(b'torch.ByteTensor', dtype=np.uint8)
add_tensor_reader(b'torch.CharTensor', dtype=np.int8)
add_tensor_reader(b'torch.ShortTensor', dtype=np.int16)
add_tensor_reader(b'torch.IntTensor', dtype=np.int32)
add_tensor_reader(b'torch.LongTensor', dtype=np.int64)
add_tensor_reader(b'torch.FloatTensor', dtype=np.float32)
add_tensor_reader(b'torch.DoubleTensor', dtype=np.float64)
add_tensor_reader(b'torch.CudaTensor', dtype=np.float32)
add_tensor_reader(b'torch.CudaByteTensor', dtype=np.uint8)
add_tensor_reader(b'torch.CudaCharTensor', dtype=np.int8)
add_tensor_reader(b'torch.CudaShortTensor', dtype=np.int16)
add_tensor_reader(b'torch.CudaIntTensor', dtype=np.int32)
add_tensor_reader(b'torch.CudaDoubleTensor', dtype=np.float64)


def add_storage_reader(typename, dtype):
    def read_storage(reader, version):
        # https://github.com/torch/torch7/blob/1e86025/generic/Storage.c#L237
        size = reader.read_long()
        return np.fromfile(reader.f, dtype=dtype, count=size)
    type_handlers[typename] = read_storage
add_storage_reader(b'torch.ByteStorage', dtype=np.uint8)
add_storage_reader(b'torch.CharStorage', dtype=np.int8)
add_storage_reader(b'torch.ShortStorage', dtype=np.int16)
add_storage_reader(b'torch.IntStorage', dtype=np.int32)
add_storage_reader(b'torch.LongStorage', dtype=np.int64)
add_storage_reader(b'torch.FloatStorage', dtype=np.float32)
add_storage_reader(b'torch.DoubleStorage', dtype=np.float64)
add_storage_reader(b'torch.CudaStorage', dtype=np.float32)
add_storage_reader(b'torch.CudaByteStorage', dtype=np.uint8)
add_storage_reader(b'torch.CudaCharStorage', dtype=np.int8)
add_storage_reader(b'torch.CudaShortStorage', dtype=np.int16)
add_storage_reader(b'torch.CudaIntStorage', dtype=np.int32)
add_storage_reader(b'torch.CudaDoubleStorage', dtype=np.float64)


def add_notimpl_reader(typename):
    def read_notimpl(reader, version):
        raise NotImplementedError('Reader not implemented for: ' + typename)
    type_handlers[typename] = read_notimpl
add_notimpl_reader(b'torch.HalfTensor')
add_notimpl_reader(b'torch.HalfStorage')
add_notimpl_reader(b'torch.CudaHalfTensor')
add_notimpl_reader(b'torch.CudaHalfStorage')


@register_handler(b'tds.Vec')
def tds_Vec_reader(reader, version):
    size = reader.read_int()
    obj = []
    _ = reader.read_obj()
    for i in range(size):
        e = reader.read_obj()
        obj.append(e)
    return obj


@register_handler(b'tds.Hash')
def tds_Hash_reader(reader, version):
    size = reader.read_int()
    obj = hashable_uniq_dict()
    _ = reader.read_obj()
    for i in range(size):
        k = reader.read_obj()
        v = reader.read_obj()
        obj[k] = v
    return obj


class T7ReaderException(Exception):
    pass


class T7Reader:

    def __init__(self,
                 fileobj,
                 use_list_heuristic=True,
                 use_int_heuristic=True,
                 utf8_decode_strings=False,
                 force_deserialize_classes=None,
                 force_8bytes_long=False):
        """
        Params:
        * `fileobj`: file object to read from, must be an actual file object
                    as it will be read by `array`, `struct`, and `numpy`. Since
                    it is only read sequentially, certain objects like pipes or
                    `sys.stdin` should work as well (untested).
        * `use_list_heuristic`: automatically turn tables with only consecutive
                                positive integral indices into lists
                                (default True)
        * `use_int_heuristic`: cast all whole floats into ints (default True)
        * `utf8_decode_strings`: decode all strings as UTF8. By default they
                                remain as byte strings. Version strings always
                                are byte strings, but this setting affects
                                class names. (default False)
        * `force_deserialize_classes`: deprecated.
        """
        self.f = fileobj
        self.objects = {}  # read objects so far

        if force_deserialize_classes is not None:
            raise DeprecationWarning(
                'force_deserialize_classes is now always '
                'forced to be true, so no longer required')
        self.use_list_heuristic = use_list_heuristic
        self.use_int_heuristic = use_int_heuristic
        self.utf8_decode_strings = utf8_decode_strings
        self.force_8bytes_long = force_8bytes_long

    def _read(self, fmt):
        sz = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(sz))

    def read_boolean(self):
        return self.read_int() == 1

    def read_int(self):
        return self._read('i')[0]

    def read_long(self):
        if self.force_8bytes_long:
            return self._read('q')[0]
        else:
            return self._read('l')[0]

    def read_long_array(self, n):
        if self.force_8bytes_long:
            lst = []
            for i in range(n):
                lst.append(self.read_long())
            return lst
        else:
            arr = array('l')
            arr.fromfile(self.f, n)
            return arr.tolist()

    def read_float(self):
        return self._read('f')[0]

    def read_double(self):
        return self._read('d')[0]

    def read_string(self, disable_utf8=False):
        size = self.read_int()
        s = self.f.read(size)
        if disable_utf8 or not self.utf8_decode_strings:
            return s
        return s.decode('utf8')

    def read_obj(self):
        typeidx = self.read_int()

        if typeidx == TYPE_NIL:
            return None

        elif typeidx == TYPE_NUMBER:
            x = self.read_double()
            # Extra checking for integral numbers:
            if self.use_int_heuristic and x.is_integer():
                return int(x)
            return x

        elif typeidx == TYPE_BOOLEAN:
            return self.read_boolean()

        elif typeidx == TYPE_STRING:
            return self.read_string()

        elif (typeidx == TYPE_TABLE or typeidx == TYPE_TORCH or
                typeidx == TYPE_FUNCTION or typeidx == TYPE_RECUR_FUNCTION or
                typeidx == LEGACY_TYPE_RECUR_FUNCTION):
            # read the object reference index
            index = self.read_int()

            # check it is loaded already
            if index in self.objects:
                return self.objects[index]

            # otherwise read it
            if (typeidx == TYPE_FUNCTION or typeidx == TYPE_RECUR_FUNCTION or
                    typeidx == LEGACY_TYPE_RECUR_FUNCTION):
                size = self.read_int()
                dumped = self.f.read(size)
                upvalues = self.read_obj()
                obj = LuaFunction(size, dumped, upvalues)
                self.objects[index] = obj
                return obj

            elif typeidx == TYPE_TORCH:
                version = self.read_string(disable_utf8=True)
                if version.startswith(b'V '):
                    version_number = int(float(version.partition(b' ')[2]))
                    class_name = self.read_string(disable_utf8=True)
                else:
                    class_name = version
                    # created before existence of versioning
                    version_number = 0
                if class_name in type_handlers:
                    # TODO: can custom readers ever be self-referential?
                    self.objects[index] = None  # FIXME: if self-referential
                    obj = type_handlers[class_name](self, version)
                    self.objects[index] = obj
                else:
                    # This must be performed in two steps to allow objects
                    # to be a property of themselves.
                    obj = TorchObject(
                        class_name, version_number=version_number)
                    self.objects[index] = obj
                    # After self.objects is populated, it's safe to read in
                    # case self-referential
                    obj._obj = self.read_obj()
                return obj

            else:  # it is a table: returns a custom dict or a list
                size = self.read_int()
                # custom hashable dict, so that it can be a key, see above
                obj = hashable_uniq_dict()
                # For checking if keys are consecutive and positive ints;
                # if so, returns a list with indices converted to 0-indices.
                key_sum = 0
                keys_natural = True
                # bugfix: obj must be registered before reading keys and vals
                self.objects[index] = obj

                for _ in range(size):
                    k = self.read_obj()
                    v = self.read_obj()
                    obj[k] = v

                    if self.use_list_heuristic:
                        if not isinstance(k, int) or k <= 0:
                            keys_natural = False
                        elif isinstance(k, int):
                            key_sum += k

                if self.use_list_heuristic:
                    # n(n+1)/2 = sum <=> consecutive and natural numbers
                    n = len(obj)
                    if keys_natural and n * (n + 1) == 2 * key_sum:
                        lst = []
                        for i in range(len(obj)):
                            elem = obj[i + 1]
                            # In case it is self-referential. This is not
                            # needed in lua torch since the tables are never
                            # modified as they are here.
                            if elem == obj:
                                elem = lst
                            lst.append(elem)
                        self.objects[index] = obj = lst

                return obj

        else:
            raise T7ReaderException(
                "unknown object type / typeidx: {}".format(typeidx))


def load(filename, **kwargs):
    """
    Loads the given t7 file using default settings; kwargs are forwarded
    to `T7Reader`.
    """
    with open(filename, 'rb') as f:
        reader = T7Reader(f, **kwargs)
        return reader.read_obj()
