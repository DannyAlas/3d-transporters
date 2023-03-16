import warnings

from pyclesperanto_prototype._tier0 import (Image,
                                            create_square_matrix_from_labelmap,
                                            execute, plugin_function)


@plugin_function
def touch_matrix_to_adjacency_matrix(
    touch_matrix: Image,
    adjacency_matrix_destination: Image = None,
    self_adjacent: bool = True,
) -> Image:
    """Takes touch matrix (which is typically just half-filled) and makes a symmetrical adjacency matrix out of it.

    Furthermore, one can define if an object is adjacent to itself (default: True).

    Parameters
    ----------
    touch_matrix : Image
    adjacency_matrix_destination : Image, optional
    self_adjacent : bool, optional
        Default: true

    Returns
    -------
    adjacency_matrix_destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_adjacencyMatrixToTouchMatrix
    """
    from .._tier1 import binary_or, set, set_where_x_equals_y, transpose_xy

    warnings.warn(
        "touch_matrix_to_adjacency_matrix is deprecated, use symmetric_maximum_matrix or symmetric_minimum_matrix or symmetric_mean_matrix instead.",
        DeprecationWarning,
    )

    temp = transpose_xy(touch_matrix)
    adjacency_matrix_destination = binary_or(
        touch_matrix, temp, adjacency_matrix_destination
    )
    if self_adjacent:
        set_where_x_equals_y(adjacency_matrix_destination, 1)

    return adjacency_matrix_destination
