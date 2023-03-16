from .._tier0 import (Image, create_like, create_matrix_from_pointlists,
                      create_none, execute, plugin_function)


@plugin_function(output_creator=create_none)
def generate_minimum_intensity_between_points_matrix(
    intensity_image: Image,
    pointlist: Image,
    touch_matrix: Image = None,
    minimum_intensity_matrix_destination: Image = None,
    num_samples: int = 10,
):
    """Determine the minimum intensity between pairs of point coordinates and
    write them in a matrix.

    Parameters
    ----------
    intensity_image: Image
        image where the intensity will be measured
    pointlist: Image
        list of coordinates
    touch_matrix: Image, optional
        if only selected pairs should be measured, use this binary matrix to confige which
    minimum_intensity_matrix_destination: Image, optional
        matrix where the results are written ito
    num_samples: int, optional
        Number of samples to take along the line for averaging, default = 10

    Returns
    -------
    average_intensity_matrix_destination
    """
    from .._tier1 import set

    if minimum_intensity_matrix_destination is None:
        minimum_intensity_matrix_destination = create_matrix_from_pointlists(
            pointlist, pointlist
        )

    if touch_matrix is None:
        touch_matrix = create_like(minimum_intensity_matrix_destination)
        set(touch_matrix, 1)

    parameters = {
        "src_touch_matrix": touch_matrix,
        "src_pointlist": pointlist,
        "src_intensity": intensity_image,
        "dst_minimum_intensity_matrix": minimum_intensity_matrix_destination,
        "num_samples": int(num_samples),
    }

    execute(
        __file__,
        "minimum_intensity_between_points_matrix_x.cl",
        "minimum_intensity_between_points_matrix",
        touch_matrix.shape,
        parameters,
    )

    return minimum_intensity_matrix_destination
