import numpy as np

from .._tier0 import (Image, create_labels_like, create_like, plugin_function,
                      pull, push)
from .._tier1 import copy, set


@plugin_function(
    categories=["label processing", "in assistant"], output_creator=create_labels_like
)
def reduce_labels_to_centroids(source: Image, destination: Image = None) -> Image:
    """Takes a label map and reduces all labels to their center spots. Label IDs stay and background will be zero.

    Parameters
    ----------
    source: Image
    destination: Image, optional

    Returns
    -------
    destination

    See Also
    --------
    ..[0] https://clij.github.io/clij2-docs/reference_reduceLabelsToCentroids
    """
    from .._tier0 import create
    from .._tier1 import (paste, set, set_column, set_ramp_x,
                          write_values_to_positions)
    from .._tier9 import centroids_of_background_and_labels

    positions = centroids_of_background_and_labels(source)
    positions_and_labels = create((positions.shape[0] + 1, positions.shape[1]))
    set_ramp_x(positions_and_labels)
    set_column(
        positions, 0, -1
    )  # prevent putting a 0 at the centroid position of the background
    paste(positions, positions_and_labels, 0, 0)
    set(destination, 0)
    write_values_to_positions(positions_and_labels, destination)

    return destination
