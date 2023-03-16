from pyclesperanto_prototype._tier0 import Image, create_like, plugin_function
from pyclesperanto_prototype._tier1 import absolute
from pyclesperanto_prototype._tier2 import subtract_images


@plugin_function(categories=["combine", "in assistant"])
def absolute_difference(
    source1: Image, source2: Image, destination: Image = None
) -> Image:
    """Determines the absolute difference pixel by pixel between two images.

    <pre>f(x, y) = |x - y| </pre>

    Parameters
    ----------
    source1 : Image
        The input image to be subtracted from.
    source2 : Image
        The input image which is subtracted.
    destination : Image, optional
        The output image  where results are written into.


    Returns
    -------
    destination

    Examples
    --------
    >>> import pyclesperanto_prototype as cle
    >>> cle.absolute_difference(source1, source2, destination)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_absoluteDifference
    """

    temp = create_like(destination)

    subtract_images(source1, source2, temp)

    return absolute(temp, destination)
