from pyclesperanto_prototype._tier0 import Image, create_like, plugin_function
from pyclesperanto_prototype._tier1 import divide_images, gaussian_blur


@plugin_function(categories=["filter", "background removal", "in assistant"])
def divide_by_gaussian_background(
    source: Image,
    destination: Image = None,
    sigma_x: float = 2,
    sigma_y: float = 2,
    sigma_z: float = 2,
) -> Image:
    """Applies Gaussian blur to the input image and divides the original by
    the result.

    Parameters
    ----------
    source : Image
    destination : Image, optional
    sigma_x : Number, optional
    sigma_y : Number, optional
    sigma_z : Number, optional

    Returns
    -------
    destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_divideByGaussianBackground
    """
    temp1 = create_like(destination)

    gaussian_blur(source, temp1, sigma_x, sigma_y, sigma_z)

    return divide_images(source, temp1, destination)
