from .._tier0 import Image, execute, plugin_function


@plugin_function
def copy(source: Image, destination: Image = None) -> Image:
    """Copies an image.

    <pre>f(x) = x</pre>

    Parameters
    ----------
    source : Image
    destination : Image, optional

    Returns
    -------
    destination

    Examples
    --------
    >>> import pyclesperanto_prototype as cle
    >>> cle.copy(source, destination)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_copy
    """

    parameters = {"dst": destination, "src": source}

    execute(
        __file__,
        "clij-opencl-kernels/kernels/copy_" + str(len(destination.shape)) + "d_x.cl",
        "copy_" + str(len(destination.shape)) + "d",
        destination.shape,
        parameters,
    )
    return destination
