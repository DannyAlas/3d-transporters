from pyclesperanto_prototype._tier0 import Image, plugin_function


@plugin_function(categories=["label measurement", "mesh", "in assistant"])
def draw_distance_mesh_between_touching_labels(
    labels: Image, distance_mesh_destination: Image = None
) -> Image:
    """Starting from a label map, draw lines between touching neighbors
    resulting in a mesh.

    The end points of the lines correspond to the centroids of the labels. The
    intensity of the lines
    corresponds to the distance between these labels (in pixels or voxels).

    Notes
    -----
    * This operation assumes input images are isotropic.

    Parameters
    ----------
    labels : Image
    distance_mesh_destination : Image, optional

    Returns
    -------
    destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_drawDistanceMeshBetweenTouchingLabels
    """
    from .._tier1 import (generate_distance_matrix, generate_touch_matrix,
                          multiply_images, touch_matrix_to_mesh)
    from .._tier9 import centroids_of_labels

    centroids = centroids_of_labels(labels)
    distance_matrix = generate_distance_matrix(centroids, centroids)
    touch_matrix = generate_touch_matrix(labels)
    touch_distance_matrix = multiply_images(touch_matrix, distance_matrix)

    from .._tier1 import set

    set(distance_mesh_destination, 0)

    distance_mesh_destination = touch_matrix_to_mesh(
        centroids, touch_distance_matrix, distance_mesh_destination
    )
    return distance_mesh_destination
