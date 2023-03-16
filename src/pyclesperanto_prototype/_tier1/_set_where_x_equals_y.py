from .._tier0 import Image, execute, plugin_function


@plugin_function
def set_where_x_equals_y(source: Image, value: float = 0) -> Image:
    """Sets all pixel values a of a given image A to a constant value v in
    case its coordinates x == y.

    Otherwise the pixel is not overwritten.
    If you want to initialize an identity transfrom matrix, set all pixels to
    0 first.

    Parameters
    ----------
    source : Image
    value : Number, optional

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_setWhereXequalsY
    """

    parameters = {"dst": source, "value": float(value)}

    execute(
        __file__,
        "clij-opencl-kernels/kernels/set_where_x_equals_y_"
        + str(len(source.shape))
        + "d_x.cl",
        "set_where_x_equals_y_" + str(len(source.shape)) + "d",
        source.shape,
        parameters,
    )
    return source
