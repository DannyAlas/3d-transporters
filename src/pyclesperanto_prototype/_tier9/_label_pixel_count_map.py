from .._tier0 import Image, plugin_function, push
from .._tier1 import replace_intensities, set_column


@plugin_function(categories=["label measurement", "map", "in assistant"])
def label_pixel_count_map(source: Image, destination: Image = None) -> Image:
    """Takes a label map, determines the number of pixels per label and
    replaces every label with the that number.

    This results in a parametric image expressing area or volume.

    Parameters
    ----------
    source : Image
    destination : Image, optional

    Returns
    -------
    destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_pixelCountMap
    """
    from .._tier9 import (push_regionprops_column,
                          statistics_of_background_and_labelled_pixels)

    regionprops = statistics_of_background_and_labelled_pixels(None, source)
    values_vector = push_regionprops_column(regionprops, "area")

    set_column(values_vector, 0, 0)

    destination = replace_intensities(source, values_vector, destination)

    return destination
