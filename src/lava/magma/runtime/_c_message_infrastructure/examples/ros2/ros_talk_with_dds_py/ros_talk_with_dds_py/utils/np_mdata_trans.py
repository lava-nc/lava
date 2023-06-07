import numpy as np
from ddsmetadata.msg import DDSMetaData


DTYPE_LIST = [np.dtype(i) for i in [np.bool_,
                                    np.byte,
                                    np.ubyte,
                                    np.short,
                                    np.ushort,
                                    np.intc,
                                    np.uintc,
                                    np.int_,
                                    np.uint,
                                    np.longlong,
                                    np.ulonglong,
                                    np.single,
                                    np.double,
                                    np.longdouble,
                                    np.cfloat,
                                    np.cdouble,
                                    np.clongdouble,
                                    np.object_,
                                    np.string_,
                                    np.unicode_,
                                    np.void]]


def metadata_to_nparray(metadata):
    ndim = metadata.nd
    shape = metadata.dims
    dims = list()
    for i in range(ndim):
        dims.append(shape[i])
    dtype = np.dtype(DTYPE_LIST[metadata.type])
    np_bytes_list = metadata.mdata
    np_bytes_array = bytearray(np_bytes_list)
    np_array = np.frombuffer(np_bytes_array, dtype)
    np_array = np_array.reshape(tuple(dims))
    return np_array


def nparray_to_metadata(np_array):
    metadata = DDSMetaData()
    metadata.nd = np_array.ndim
    metadata.type = np_array.dtype.num
    metadata.elsize = np_array.itemsize
    metadata.total_size = np_array.size
    shape_list = list(np_array.shape)
    for _ in range(5 - len(shape_list)):
        shape_list.append(0)
    metadata.dims = shape_list
    strides_list = list(np_array.strides)
    for i in range(5 - len(strides_list)):
        strides_list.append(0)
    metadata.strides = strides_list
    np_bytes = np_array.tobytes()
    np_bytes_array = np.frombuffer(np_bytes, np.byte)
    np_bytes_list = np_bytes_array.tolist()
    metadata.mdata = np_bytes_list
    return metadata
