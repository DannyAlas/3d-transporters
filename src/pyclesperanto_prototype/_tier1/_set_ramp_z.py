from .._tier0 import Image, execute, plugin_function


@plugin_function
def set_ramp_z(source: Image) -> Image:
    """Sets all pixel values to their Z coordinate

    Parameters
    ----------
    source : Image

    Examples
    --------
    >>> import pyclesperanto_prototype as cle
    >>> cle.set_ramp_z(source)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_setRampZ
    """

    parameters = {"dst": source}

    execute(
        __file__,
        "clij-opencl-kernels/kernels/set_ramp_z_" + str(len(source.shape)) + "d_x.cl",
        "set_ramp_z_" + str(len(source.shape)) + "d",
        source.shape,
        parameters,
    )
    return source
