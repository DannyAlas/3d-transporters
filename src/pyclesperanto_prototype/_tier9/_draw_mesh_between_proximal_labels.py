from .._tier0 import Image, plugin_function
from .._tier1 import (generate_distance_matrix, n_closest_points,
                      point_index_list_to_mesh, set, set_column, set_row,
                      set_where_x_equals_y)
from .._tier9 import centroids_of_labels


@plugin_function(categories=["label measurement", "mesh", "in assistant"])
def draw_mesh_between_proximal_labels(
    labels: Image, mesh_target: Image = None, maximum_distance: int = 1
) -> Image:
    """Starting from a label map, draw lines between labels that are closer
    than a given distance resulting in a mesh.

    The end points of the lines correspond to the centroids of the labels.

    Notes
    -----
    * This operation assumes input images are isotropic.

    Parameters
    ----------
    labels : Image
    mesh_target : Image, optional
    maximum_distance : Number, optional

    Returns
    -------
    destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_drawMeshBetweenProximalLabels
    """
    pointlist = centroids_of_labels(labels)

    distance_matrix = generate_distance_matrix(pointlist, pointlist)

    from .._tier1 import set
    from .._tier2 import distance_matrix_to_mesh

    set(mesh_target, 0)

    mesh_target = distance_matrix_to_mesh(
        pointlist, distance_matrix, mesh_target, maximum_distance
    )
    return mesh_target
