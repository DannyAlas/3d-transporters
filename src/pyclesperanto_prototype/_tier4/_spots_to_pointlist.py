from .._tier0 import Image, create, create_none, plugin_function
from .._tier2 import label_spots, maximum_of_all_pixels
from .._tier3 import labelled_spots_to_pointlist


@plugin_function(output_creator=create_none)
def spots_to_pointlist(
    input_spots: Image, destination_pointlist: Image = None
) -> Image:
    """Transforms a spots image as resulting from maximum/minimum detection in
    an image where every column contains d
    pixels (with d = dimensionality of the original image) with the coordinates of
    the maxima/minima.

    Parameters
    ----------
    input_spots : Image
    destination_pointlist : Image, optional

    Returns
    -------
    destination_pointlist

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_spotsToPointList
    """
    labelled_spots = label_spots(input_spots)

    destination_pointlist = labelled_spots_to_pointlist(
        labelled_spots, destination_pointlist
    )

    return destination_pointlist
