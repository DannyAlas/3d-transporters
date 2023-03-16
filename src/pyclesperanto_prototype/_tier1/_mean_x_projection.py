from .._tier0 import Image, create_2d_yz, execute, plugin_function


@plugin_function(output_creator=create_2d_yz, categories=["projection"])
def mean_x_projection(source: Image, destination: Image = None) -> Image:
    """Determines the mean average intensity projection of an image along X.

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
    >>> cle.mean_x_projection(source, destination)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_meanXProjection
    """

    parameters = {
        "dst": destination,
        "src": source,
    }

    execute(
        __file__,
        "clij-opencl-kernels/kernels/mean_x_projection_x.cl",
        "mean_x_projection",
        destination.shape,
        parameters,
    )
    return destination
