from .._tier0 import Image, create_binary_like, execute, plugin_function


@plugin_function(
    categories=["binarize", "in assistant"], output_creator=create_binary_like
)
def detect_maxima_box(
    source: Image,
    destination: Image = None,
    radius_x: int = 0,
    radius_y: int = 0,
    radius_z: int = 0,
) -> Image:
    """Detects local maxima in a given square/cubic neighborhood.

    Pixels in the resulting image are set to 1 if there is no other pixel in a
    given radius which has a
    higher intensity, and to 0 otherwise.

    Parameters
    ----------
    source : Image
    destination : Image, optional
    radius_x : Number, optional
    radius_y : Number, optional
    radius_z : Number, optional

    Returns
    -------
    destination

    Examples
    --------
    >>> import pyclesperanto_prototype as cle
    >>> cle.detect_maxima_box(source, destination, 1, 1, 1)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_detectMaximaBox
    """

    from .._tier0 import create_like
    from .._tier1 import mean_box

    temp = create_like(source)
    mean_box(source, temp, radius_x, radius_y, radius_z)

    parameters = {"src": temp, "dst": destination}

    # todo: ensure detect_maxima_2d_x.cl fit to naming convention
    execute(
        __file__,
        "detect_maxima_" + str(len(destination.shape)) + "d_x.cl",
        "detect_maxima_" + str(len(destination.shape)) + "d",
        destination.shape,
        parameters,
    )

    return destination
