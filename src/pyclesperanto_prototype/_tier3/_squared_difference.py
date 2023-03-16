from pyclesperanto_prototype._tier0 import Image, create_like, plugin_function
from pyclesperanto_prototype._tier1 import power
from pyclesperanto_prototype._tier2 import subtract_images


@plugin_function(categories=["combine", "in assistant"])
def squared_difference(
    source1: Image, source2: Image, destination: Image = None
) -> Image:
    """Determines the squared difference pixel by pixel between two images.

    Parameters
    ----------
    source1 : Image
    source2 : Image
    destination : Image, optional

    Returns
    -------
    destination

    Examples
    --------
    >>> import pyclesperanto_prototype as cle
    >>> cle.squared_difference(source1, source2, destination)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_squaredDifference
    """

    temp = create_like(destination)

    subtract_images(source1, source2, temp)

    return power(temp, destination, 2)
