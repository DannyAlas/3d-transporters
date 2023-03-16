from pyclesperanto_prototype._tier0 import Image, plugin_function


@plugin_function(categories=["label measurement", "mesh", "in assistant"])
def draw_mesh_between_touching_labels(
    labels: Image, distance_mesh_destination: Image = None
) -> Image:
    """Starting from a label map, draw lines between touching neighbors
    resulting in a mesh.

    The end points of the lines correspond to the centroids of the labels.

    Parameters
    ----------
    labels : Image
    distance_mesh_destination : Image, optional

    Returns
    -------
    destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_drawMeshBetweenTouchingLabels
    """
    from .._tier1 import generate_touch_matrix, touch_matrix_to_mesh
    from .._tier9 import centroids_of_labels

    centroids = centroids_of_labels(labels)
    touch_matrix = generate_touch_matrix(labels)

    from .._tier1 import set

    set(distance_mesh_destination, 0)
    touch_mesh_destination = touch_matrix_to_mesh(
        centroids, touch_matrix, distance_mesh_destination
    )
    return touch_mesh_destination
