import numpy as np

from .._tier0 import (Image, create_labels_like, create_like, plugin_function,
                      pull, push)
from .._tier1 import (copy, onlyzero_overwrite_maximum_box,
                      onlyzero_overwrite_maximum_diamond, set)


@plugin_function(
    categories=["label processing", "in assistant"], output_creator=create_labels_like
)
def dilate_labels(
    labeling_source: Image, labeling_destination: Image = None, radius: int = 2
) -> Image:
    """Dilates labels to a larger size. No label overwrites another label.
    Similar to the implementation in scikit-image [2] and MorpholibJ[3]

    Notes
    -----
    * This operation assumes input images are isotropic.

    Parameters
    ----------
    labels_input : Image
        label image to erode
    labels_destination : Image, optional, optional
        result
    radius : int, optional

    Returns
    -------
    labels_destination

    See Also
    --------
    ..[1] https://clij.github.io/clij2-docs/reference_dilateLabels
    ..[2] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_expand_labels.html?highlight=expand%20labels
    ..[3] https://github.com/ijpb/MorphoLibJ
    """
    flip = create_like(labeling_destination)
    flop = create_like(labeling_destination)

    flag = push(np.asarray([[[0]]]))
    flag_value = 1

    copy(labeling_source, flip)

    iteration_count = 0

    while flag_value > 0 and iteration_count < radius:
        if iteration_count % 2 == 0:
            onlyzero_overwrite_maximum_box(flip, flag, flop)
        else:
            onlyzero_overwrite_maximum_diamond(flop, flag, flip)
        flag_value = pull(flag)[0][0][0]
        set(flag, 0)
        iteration_count += 1

    if iteration_count % 2 == 0:
        copy(flip, labeling_destination)
    else:
        copy(flop, labeling_destination)

    return labeling_destination
