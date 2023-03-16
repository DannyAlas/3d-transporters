from .._tier0 import Image, create_like, execute, plugin_function
from .._tier1 import greater_constant, multiply_matrix


@plugin_function
def neighbors_of_neighbors(
    touch_matrix: Image, neighbor_matrix_destination: Image = None
) -> Image:
    """Determines neighbors of neigbors from touch matrix and saves the result
    as a new touch matrix.

    Parameters
    ----------
    touch_matrix : Image
    neighbor_matrix_destination : Image, optional

    Returns
    -------
    neighbor_matrix_destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_neighborsOfNeighbors
    """
    from .._tier2 import touch_matrix_to_adjacency_matrix

    touch_matrix = touch_matrix_to_adjacency_matrix(touch_matrix)

    temp = create_like(touch_matrix)
    multiply_matrix(touch_matrix, touch_matrix, temp)

    neighbor_matrix_destination = greater_constant(temp, neighbor_matrix_destination, 0)

    return neighbor_matrix_destination
