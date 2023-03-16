from typing import Union

import numpy as np
import pyopencl as cl

from ._pycl import OCLArray, _OCLImage

Image = Union[np.ndarray, OCLArray, cl.Image, _OCLImage]

def is_image(object):
    return isinstance(object, np.ndarray) or \
           isinstance(object, tuple) or \
           isinstance(object, list) or \
           isinstance(object, OCLArray) or \
           str(type(object)) in ["<class 'cupy._core.core.ndarray'>",
                                 "<class 'dask.array.core.Array'>",
                                 "<class 'xarray.core.dataarray.DataArray'>",
                                 "<class 'resource_backed_dask_array.ResourceBackedDaskArray'>",
                                 "<class 'torch.Tensor'>"]
