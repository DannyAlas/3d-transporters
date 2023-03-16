from pyclesperanto_prototype._tier0 import (Image,
                                            create_square_matrix_from_labelmap,
                                            execute, plugin_function)


@plugin_function
def symmetric_minimum_matrix(
    source_matrix: Image, destination_matrix: Image = None
) -> Image:
    """Takes matrix (which might be asymmetric) and makes a symmetrical matrix out of it by taking the minimum value
    of m(x,y) and m(y,x) and storing it in both entries.

    Parameters
    ----------
    source_matrix : Image
    destination_matrix : Image, optional

    Returns
    -------
    destination_matrix
    """
    from .._tier1 import minimum_images, transpose_xy

    temp = transpose_xy(source_matrix)
    destination_matrix = minimum_images(source_matrix, temp, destination_matrix)

    return destination_matrix
