import numpy as np

from .._tier0 import Image, execute, plugin_function


@plugin_function(categories=["combine", "neighbor", "map", "in assistant"])
def standard_deviation_of_n_nearest_neighbors_map(
    parametric_map: Image,
    label_map: Image,
    parametric_map_destination: Image = None,
    n: int = 1,
) -> Image:
    """Takes a label image and a parametric intensity image and will replace each labels value in the parametric image
    by the standard_deviation value of neighboring labels. The number of nearest neighbors can be configured.

    Notes
    -----
    * Values of all pixels in a label each must be identical.
    * This operation assumes input images are isotropic.

    Parameters
    ----------
    parametric_map : Image
    label_map : Image
    parametric_map_destination : Image, optional
    n : int
        number of nearest neighbors

    Returns
    -------
    parametric_map_destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_standardDeviationOfNNearestNeighbors
    """
    from .._tier1 import (generate_distance_matrix, read_intensities_from_map,
                          replace_intensities)
    from .._tier2 import standard_deviation_of_touching_neighbors
    from .._tier3 import generate_n_nearest_neighbors_matrix
    from .._tier9 import centroids_of_labels

    centroids = centroids_of_labels(label_map)

    distance_matrix = generate_distance_matrix(centroids, centroids)

    touch_matrix = generate_n_nearest_neighbors_matrix(distance_matrix, n=n)

    intensities = read_intensities_from_map(label_map, parametric_map)

    new_intensities = standard_deviation_of_touching_neighbors(
        intensities, touch_matrix
    )

    parametric_map_destination = replace_intensities(
        label_map, new_intensities, parametric_map_destination
    )

    return parametric_map_destination
