from .._tier0 import Image, create, create_from_pointlist, plugin_function
from .._tier1 import (add_image_and_scalar, add_images_weighted, paste,
                      set_ramp_x, write_values_to_positions)


@plugin_function(output_creator=create_from_pointlist)
def pointlist_to_labelled_spots(
    pointlist: Image, spots_destination: Image = None
) -> Image:
    """Takes a pointlist with dimensions n times d with n point coordinates in
    d dimensions and labels corresponding pixels.

    Parameters
    ----------
    pointlist : Image
    spots_destination : Image, optional

    Returns
    -------
    spots_destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_pointlistToLabelledSpots
    """

    temp1 = create([pointlist.shape[0] + 1, pointlist.shape[1]])
    temp2 = create([pointlist.shape[0] + 1, pointlist.shape[1]])

    set_ramp_x(temp1)
    add_image_and_scalar(temp1, temp2, 1)
    paste(pointlist, temp2)
    from .._tier1 import set

    set(spots_destination, 0)
    spots_destination = write_values_to_positions(temp2, spots_destination)

    return spots_destination
