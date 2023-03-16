from ._available_device_names import available_device_names
from ._backends import Backend
from ._categories import categories
from ._cl_image import create_image, empty_image, empty_image_like
from ._cl_info import cl_info
from ._create import (create, create_2d_xy, create_2d_xz, create_2d_yx,
                      create_2d_yz, create_2d_zx, create_2d_zy,
                      create_binary_like, create_from_pointlist,
                      create_labels_like, create_like,
                      create_matrix_from_pointlists, create_none,
                      create_pointlist_from_labelmap,
                      create_square_matrix_from_labelmap,
                      create_square_matrix_from_pointlist,
                      create_square_matrix_from_two_labelmaps,
                      create_vector_from_labelmap,
                      create_vector_from_square_matrix, create_zyx)
from ._device import get_device, select_device, set_device_scoring_key
from ._execute import execute
from ._operations import operation, operations, search_operation_names
from ._plugin_function import plugin_function
from ._pull import pull
from ._pull import pull as nparray
from ._pull import pull_zyx
from ._push import push
from ._push import push as asarray
from ._push import push_zyx
from ._radius_to_kernel_size import radius_to_kernel_size
from ._set_wait_for_kernel_finish import set_wait_for_kernel_finish
from ._sigma_to_kernel_size import sigma_to_kernel_size
from ._types import Image


def _warn_of_interpolation_not_available():
    import warnings

    warnings.warn(
        "Creating an OpenCL image failed on the GPU. Hence, linear interpolation is not possible.\n"
        "Nearest-neighbor interpolation will be used instead."
    )


# Prevent an issue with Intel GPUs on MacOS (pre M1/M2)
from sys import platform

if platform == "darwin":
    device_name = str(get_device())
    if "Intel" in device_name:
        select_device("CPU")
