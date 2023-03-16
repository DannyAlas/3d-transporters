from .. import binary_or, greater_constant, smaller_constant
from .._tier0 import Image, create_like, create_none, plugin_function


@plugin_function(output_creator=create_none, categories=["label processing", "combine"])
def exclude_labels_with_values_out_of_range(
    values_vector: Image,
    label_map_input: Image,
    label_map_destination: Image = None,
    minimum_value_range: float = 0,
    maximum_value_range: float = 100,
) -> Image:
    """This operation removes labels from a labelmap and renumbers the
    remaining labels.

    Hand over a vector of values and a range specifying which labels with which
    values are eliminated.

    Parameters
    ----------
    values_vector : Image
    label_map_input : Image
    label_map_destination : Image, optional
    minimum_value_range : Number, optional
    maximum_value_range : Number, optional

    Returns
    -------
    label_map_destination

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_excludeLabelsWithValuesOutOfRange
    """
    above = create_like(values_vector)
    below = create_like(values_vector)
    flaglist_vector = create_like(values_vector)

    smaller_constant(values_vector, below, minimum_value_range)
    greater_constant(values_vector, above, maximum_value_range)

    binary_or(below, above, flaglist_vector)

    from .._tier3 import exclude_labels

    label_map_destination = exclude_labels(
        flaglist_vector, label_map_input, label_map_destination
    )

    return label_map_destination
