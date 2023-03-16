from .._tier0 import Image, plugin_function, push
from .._tier1 import replace_intensities, set_column


@plugin_function(
    categories=["combine", "label measurement", "map", "in assistant"], priority=-1
)
def label_mean_intensity_map(
    source: Image, label_map: Image, destination: Image = None
) -> Image:
    """Takes an image and a corresponding label map, determines the mean
    intensity per label and replaces every label with the that number.

    This results in a parametric image expressing mean object intensity.

    Parameters
    ----------
    source : Image
    label_map : Image
    destination : Image, optional

    Returns
    -------
    destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_meanIntensityMap
    """
    from .._tier9 import (push_regionprops_column,
                          statistics_of_background_and_labelled_pixels)

    regionprops = statistics_of_background_and_labelled_pixels(source, label_map)

    values_vector = push_regionprops_column(regionprops, "mean_intensity")
    set_column(values_vector, 0, 0)

    destination = replace_intensities(label_map, values_vector, destination)

    return destination
