from .._tier0 import (Image, create_2d_yz, execute, plugin_function,
                      radius_to_kernel_size)


@plugin_function(output_creator=create_2d_yz, categories=["projection"])
def sum_x_projection(source: Image, destination: Image = None) -> Image:
    """Determines the sum intensity projection of an image along Z.

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
    >>> cle.sum_x_projection(source, destination)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_sumXProjection
    """

    parameters = {
        "dst": destination,
        "src": source,
    }

    execute(
        __file__,
        "clij-opencl-kernels/kernels/sum_x_projection_x.cl",
        "sum_x_projection",
        destination.shape,
        parameters,
    )
    return destination
