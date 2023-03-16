from .._tier0 import Image, create_none, plugin_function


@plugin_function(output_creator=create_none)
def centroids_of_background_and_labels(
    source: Image, pointlist_destination: Image = None
) -> Image:
    """See centroids_of_labels"""
    from .._tier9 import centroids_of_labels

    return centroids_of_labels(source, pointlist_destination, True)
