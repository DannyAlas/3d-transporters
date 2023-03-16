from .._tier0 import Image, create, create_labels_like, plugin_function
from .._tier1 import replace_intensities, set, set_column
from .._tier2 import (block_enumerate, flag_existing_intensities,
                      maximum_of_all_pixels, sum_reduction_x)


@plugin_function(
    output_creator=create_labels_like, categories=["label processing", "in assistant"]
)
def relabel_sequential(
    source: Image, output: Image = None, blocksize: int = 4096
) -> Image:
    """Analyses a label map and if there are gaps in the indexing (e.g. label
    5 is not present) all
    subsequent labels will be relabelled.

    Thus, afterwards number of labels and maximum label index are equal.
    This operation is mostly performed on the CPU.

    Parameters
    ----------
    labeling_input : Image
    labeling_destination : Image, optional
    blocksize : int, optional
        Renumbering is done in blocks for performance reasons.
        Change the blocksize to adapt to your data and hardware

    Returns
    -------
    labeling_destination

    Examples
    --------
    >>> import pyclesperanto_prototype as cle
    >>> cle.relabel_sequential(labeling_input, labeling_destination)

    References
    ----------
    .. [1] https://clij.github.io/clij2-docs/reference_closeIndexGapsInLabelMap
    """
    max_label = maximum_of_all_pixels(source)

    flagged_indices = create([1, int(max_label) + 1])
    set(flagged_indices, 0)
    flag_existing_intensities(source, flagged_indices)
    set_column(flagged_indices, 0, 0)  # background shouldn't be relabelled

    # sum existing labels per blocks
    block_sums = create([1, int((int(max_label) + 1) / blocksize) + 1])
    sum_reduction_x(flagged_indices, block_sums, blocksize)

    # distribute new numbers
    new_indices = create([1, int(max_label) + 1])
    block_enumerate(flagged_indices, block_sums, new_indices, blocksize)

    replace_intensities(source, new_indices, output)

    return output
