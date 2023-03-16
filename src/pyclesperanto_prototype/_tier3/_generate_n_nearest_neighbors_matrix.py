from .._tier0 import (Image, create_square_matrix_from_labelmap, execute,
                      plugin_function)


@plugin_function
def generate_n_nearest_neighbors_matrix(
    distance_matrix: Image, touch_matrix_destination: Image = None, n: int = 1
) -> Image:
    """Produces a touch-matrix where the n nearest neighbors are marked as touching neighbors.

    Takes a distance matrix (e.g. derived from a pointlist of centroids) and marks for every column the n smallest
    distances as neighbors. The resulting matrix can be use as if it was a touch-matrix (a.k.a. adjacency graph matrix).

    Inspired by a similar implementation in imglib2 [1]

    Note: The implementation is limited to square matrices.

    Parameters
    ----------
    distance_marix : Image
    touch_matrix_destination : Image, optional
    n : int, optional
        number of neighbors

    References
    ----------
    [1] https://github.com/imglib/imglib2/blob/master/src/main/java/net/imglib2/interpolation/neighborsearch/InverseDistanceWeightingInterpolator.java

    Returns
    -------
    touch_matrix_destination
    """
    import numpy as np

    from .._tier1 import (copy, n_closest_points,
                          point_index_list_to_touch_matrix, set, set_column,
                          set_row, set_where_x_equals_y)

    distance_matrix = copy(distance_matrix)

    # ignore background and ignore self
    set_row(distance_matrix, 0, np.finfo(np.float32).max)
    set_column(distance_matrix, 0, np.finfo(np.float32).max)
    set_where_x_equals_y(distance_matrix, np.finfo(np.float32).max)

    index_list = n_closest_points(distance_matrix, n=n)

    set(touch_matrix_destination, 0)

    touch_matrix_destination = point_index_list_to_touch_matrix(
        index_list, touch_matrix_destination
    )

    set_column(touch_matrix_destination, 0, 0)  # no label touches the background

    return touch_matrix_destination
