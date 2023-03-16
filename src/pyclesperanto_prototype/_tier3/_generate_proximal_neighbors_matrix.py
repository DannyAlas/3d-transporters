import numpy as np

from .._tier0 import (Image, create_square_matrix_from_labelmap, execute,
                      plugin_function)


@plugin_function
def generate_proximal_neighbors_matrix(
    distance_matrix: Image,
    touch_matrix_destination: Image = None,
    min_distance: float = 0,
    max_distance: float = np.finfo(np.float32).max,
):
    """Produces a touch-matrix where the neighbors within a given distance range are marked as touching neighbors.

    Takes a distance matrix (e.g. derived from a pointlist of centroids) and marks for every column the neighbors whose
    distance lie within a given distance range (>= min and <= max).
    The resulting matrix can be use as if it was a touch-matrix (a.k.a. adjacency graph matrix).

    Parameters
    ----------
    distance_matrix : Image
    touch_matrix_destination : Image, optional
    min_distance : float, optional, optional
        default : 0
    max_distance : float, optional, optional
        default: maximum float value

    Returns
    -------
    touch_matrix_destination
    """
    from .._tier1 import (binary_and, greater_or_equal_constant, set_column,
                          set_row, set_where_x_equals_y,
                          smaller_or_equal_constant)

    above_min_distance = greater_or_equal_constant(
        distance_matrix, constant=min_distance
    )
    below_max_distance = smaller_or_equal_constant(
        distance_matrix, constant=max_distance
    )
    touch_matrix_destination = binary_and(
        above_min_distance, below_max_distance, touch_matrix_destination
    )

    # no label touches the background or itself
    set_where_x_equals_y(touch_matrix_destination, 0)
    set_row(touch_matrix_destination, 0, 0)
    set_column(touch_matrix_destination, 0, 0)

    return touch_matrix_destination
