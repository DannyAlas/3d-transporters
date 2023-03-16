from .._tier0 import Image, plugin_function, push
from .._tier1 import replace_intensities, set_column


@plugin_function(categories=["label measurement", "map", "in assistant"])
def label_maximum_extension_map(labels: Image, destination: Image = None) -> Image:
    """Takes a label map, determines for every label the maximum distance
    of any pixel to the centroid and replaces every label with the that number.

    Parameters
    ----------
    labels : Image
    destination : Image, optional

    Returns
    -------
    destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_maximumExtensionMap
    """
    from .._tier9 import (push_regionprops_column,
                          statistics_of_background_and_labelled_pixels)

    regionprops = statistics_of_background_and_labelled_pixels(None, labels)
    values_vector = push_regionprops_column(regionprops, "max_distance_to_centroid")

    set_column(values_vector, 0, 0)

    destination = replace_intensities(labels, values_vector, destination)

    return destination
