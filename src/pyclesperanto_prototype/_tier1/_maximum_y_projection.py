from .._tier0 import Image, create_2d_zx, execute, plugin_function


@plugin_function(output_creator=create_2d_zx, categories=["projection"])
def maximum_y_projection(source: Image, destination_max: Image = None) -> Image:
    """Determines the maximum intensity projection of an image along X.

    Parameters
    ----------
    source : Image
    destination_max : Image, optional

    Returns
    -------
    destination_max

    Examples
    --------
    >>> import pyclesperanto_prototype as cle
    >>> cle.maximum_y_projection(source, destination_max)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_maximumYProjection
    """

    parameters = {
        "dst_max": destination_max,
        "src": source,
    }

    execute(
        __file__,
        "clij-opencl-kernels/kernels/maximum_y_projection_x.cl",
        "maximum_y_projection",
        destination_max.shape,
        parameters,
    )
    return destination_max
