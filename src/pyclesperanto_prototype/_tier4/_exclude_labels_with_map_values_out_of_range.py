from .._tier0 import Image, create_labels_like, plugin_function


@plugin_function(
    output_creator=create_labels_like,
    categories=["label processing", "combine", "in assistant"],
)
def exclude_labels_with_map_values_out_of_range(
    values_map: Image,
    label_map_input: Image,
    label_map_destination: Image = None,
    minimum_value_range: float = 0,
    maximum_value_range: float = 100,
) -> Image:
    """This operation removes labels from a labelmap and renumbers the
    remaining labels.

    Notes
    -----
    * Values of all pixels in a label each must be identical.

    Parameters
    ----------
    values_map : Image
    label_map_input : Image
    label_map_destination : Image, optional
    minimum_value_range : Number, optional
    maximum_value_range : Number, optional

    Returns
    -------
    label_map_destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_excludeLabelsWithValuesWithinRange
    """
    from .._tier1 import read_intensities_from_map

    values_vector = read_intensities_from_map(label_map_input, values_map)
    from .._tier3 import exclude_labels_with_values_out_of_range

    return exclude_labels_with_values_out_of_range(
        values_vector,
        label_map_input,
        label_map_destination,
        minimum_value_range,
        maximum_value_range,
    )
