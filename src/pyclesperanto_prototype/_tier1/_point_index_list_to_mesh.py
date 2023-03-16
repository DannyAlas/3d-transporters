from .._tier0 import Image, create_from_pointlist, execute, plugin_function


@plugin_function(output_creator=create_from_pointlist)
def point_index_list_to_mesh(
    pointlist: Image, indexlist: Image, mesh_destination: Image = None
) -> Image:
    """Meshes all points in a given point list which are indiced in a
    corresponding index list.

    Parameters
    ----------
    pointlist : Image
    indexlist : Image
    mesh_destination : Image, optional

    Returns
    -------
    mesh_destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_pointIndexListToMesh
    """
    from .._tier1 import set

    set(mesh_destination, 0)

    parameters = {
        "src_pointlist": pointlist,
        "src_indexlist": indexlist,
        "dst_mesh": mesh_destination,
    }

    dimensions = [1, 1, indexlist.shape[-1]]

    execute(
        __file__,
        "pointindexlist_to_mesh_3d_x.cl",
        "pointindexlist_to_mesh_3d",
        dimensions,
        parameters,
    )

    return mesh_destination
