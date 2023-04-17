import logging
import os
import warnings
from asyncio.log import logger
from multiprocessing import Pool, cpu_count
from typing import Dict, NewType, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from napari import Viewer
from napari.types import ImageData, LabelsData
from nd2reader import ND2Reader
import scipy
from skimage.exposure import rescale_intensity
from skimage.filters import (gaussian, threshold_isodata, threshold_li,
                             threshold_mean, threshold_minimum, threshold_otsu,
                             threshold_triangle, threshold_yen)
from skimage.io import imsave
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import local_maxima
from skimage.segmentation import watershed as _watershed
from skimage.util import img_as_ubyte

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.handlers = []

THRESH_METHODS = {
    "Isodata": threshold_isodata,
    "Li": threshold_li,
    "Mean": threshold_mean,
    "Minimum": threshold_minimum,
    "Otsu": threshold_otsu,
    "Triangle": threshold_triangle,
    "Yen": threshold_yen,
}

FAILED = {}


class _layers:
    def __iter__(self):
        for attr, value in self.__dict__.items():
            if not attr.startswith("__") and not callable(attr):
                yield value

    @property
    def layers(self):
        return [layer for layer in self]

    def add_layer(self, data, name="layer"):
        setattr(self, name, _layer(name, data))

    def remove_layer(self, name):
        delattr(self, name)


class _layer:
    def __init__(self, name, data):
        self.name = name
        self.data = data


def voronoi_otsu_labeling(
    image: ImageData, spot_sigma: float = 2, outline_sigma: float = 2
) -> LabelsData:
    """Voronoi-Otsu-Labeling is a segmentation algorithm for blob-like structures such as nuclei and granules with high signal intensity on low-intensity background.

    Parameters
    ----------
    image : ndarray
        Input image.
    spot_sigma : float, optional
        Standard deviation of the Gaussian kernel used to smooth the image. Controls how close detected cells can be
    outline_sigma : float, optional
        Standard deviation of the Gaussian kernel used to smooth the outlines. Controls how precise segmented objects are.

    Returns
    -------
    labels : ndarray
        Labeled image.

    References
    ----------
    .. [1] https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb
    """
    image = np.asarray(image)

    # blur and detect local maxima
    blurred_spots = gaussian(image, spot_sigma)
    spot_centroids = local_maxima(blurred_spots)

    # blur and threshold
    blurred_outline = gaussian(image, outline_sigma)
    threshold = threshold_otsu(blurred_outline)
    binary_otsu = blurred_outline > threshold

    # determine local maxima within the thresholded area
    remaining_spots = spot_centroids * binary_otsu

    # start from remaining spots and flood binary image with labels
    labeled_spots = label(remaining_spots)
    labels = _watershed(binary_otsu, labeled_spots, mask=binary_otsu)

    return labels


def set_napari_channels(viewer: Viewer, metadata):
    for i, channel in enumerate(metadata["channels"]):
        logger.info(f"Channel: {channel} ({i})")
        # rename TRITC to VGaT
        if channel == "TRITC":
            channel = "VGaT"
        # rename CY5 to VGLUT3
        if channel == "Cy5":
            channel = "VGluT3"
        # create a variable for each channel
        exec(f"{channel} = viewer.layers[{i}].data")
        logger.debug(
            f"Channel: {channel} ({i}) - {viewer.layers[i].data.shape} - {viewer.layers[i].data.dtype} - {viewer.layers[i].data.nbytes} bytes"
        )
        # rename the layer
        viewer.layers[i].name = channel

    # set the color of layers
    for i, layer in enumerate(viewer.layers):
        if layer.name == "DAPI":
            layer.colormap = "blue"
        elif layer.name == "EGFP":
            layer.colormap = "green"
        elif layer.name == "VGaT":
            layer.colormap = "red"
        elif layer.name == "VGluT3":
            layer.colormap = "magenta"


def save_plt(file, name, plt: plt, folder: str = None):
    # get the file name
    file_name = os.path.basename(file)
    # check if a folder exists for the file
    if not os.path.exists(os.path.join(os.path.dirname(file), file_name.split(".")[0])):
        os.mkdir(os.path.join(os.path.dirname(file), file_name.split(".")[0]))
    # check if a folder is given
    if folder:
        logger.debug(f"Folder: {folder}")
        # check if the folder exists
        if not os.path.exists(
            os.path.join(os.path.dirname(file), file_name.split(".")[0], folder)
        ):
            os.mkdir(
                os.path.join(os.path.dirname(file), file_name.split(".")[0], folder)
            )
            logger.debug(
                f'Folder created: {os.path.join(os.path.dirname(file), file_name.split(".")[0], folder)}'
            )
        # save the plot
        plt.savefig(
            os.path.join(
                os.path.dirname(file), file_name.split(".")[0], folder, name + ".png"
            )
        )
        logger.info(
            f'Plot saved: {os.path.join(os.path.dirname(file), file_name.split(".")[0], folder, name + ".png")}'
        )
    else:
        # save the plot
        plt.savefig(
            os.path.join(os.path.dirname(file), file_name.split(".")[0], name + ".png")
        )
        logger.info(
            f'Plot saved: {os.path.join(os.path.dirname(file), file_name.split(".")[0], name + ".png")}'
        )


def increase_contrast(
    image: ImageData, low: float = 0.5, high: float = 99.5
) -> ImageData:
    """Increase the contrast of an image by rescaling the intensity to the a low and high percentile.

    Parameters
    ----------
    image : ndarray
        Input image.
    low : float, optional
        Low percentile, by default 0.5
    high : float, optional
        High percentile, by default 99.5
    """
    percentiles = np.percentile(image, (low, high))
    return rescale_intensity(image, in_range=tuple(percentiles))


def create_threshold_hist_plot(image: np.ndarray, file):
    """Create a plot with a histogram of the image and the threshold of different thresholding THRESH_METHODS."""

    logger.info(f"Creating threshold histogram plot")
    logger.debug(f"Image shape: {image.shape}")

    if len(image.shape) > 2:
        raise ValueError("Image must be 2 dimensional.")
    threshes = {}
    fig, axes = plt.subplots(2, len(THRESH_METHODS), figsize=(20, 5))
    try:
        for i, (method, func) in enumerate(THRESH_METHODS.items()):
            axes[0, i].imshow(image > func(image), cmap="gray")
            axes[0, i].set_title(method)
            axes[0, i].axis("off")
            axes[1, i].hist(image.ravel(), bins=256)
            thresh = func(image)
            axes[1, i].axvline(thresh, color="r")
            axes[1, i].set_title(f"{method} threshold: {int(thresh)}")
            # get the amount of pixels above and below the threshold
            above_thresh = np.sum(image > thresh)
            below_thresh = np.sum(image < thresh)
            threshes[method] = {"above": above_thresh, "below": below_thresh}
        fig.suptitle("Thresholding methods", fontsize=16)
    except Exception as e:
        try:
            logger.error(
                f"Failed to create threshold histogram plot because {e}. Trying skimage's try_all_threshold"
            )
            from skimage.filters import try_all_threshold

            fig, axes = try_all_threshold(image, figsize=(20, 5), verbose=False)
            fig.suptitle("Thresholding methods", fontsize=16)
        except Exception as e:
            logger.error(f"Failed try_all_thresholds: {e}")
            if FAILED.get(str(file)):
                FAILED[str(file)]["threshold_hist"] = e
            else:
                FAILED[str(file)] = {"threshold_hist": e}

    # FIXME: implement some way to determine the best thresholding method, currently we'll just use Otsu
    # try: DOI:10.1109/ICSIPA.2009.5478623 or maybe Median & Median Absolute Deviation...

    fig.tight_layout(h_pad=0.2, w_pad=0.1)

    return fig, "Otsu"


layers = NewType("layers", Dict[str, np.ndarray])


class layers:
    """
    A class used to represent layers in the napari viewer
    """

    name = ""
    data = None


def create_threshold_plot(viewer: _layers, file: str):
    for layer in viewer:
        try:
            if len(layer.data.shape) == 3:
                tresh_img = layer.data[0]
            elif len(layer.data.shape) == 2:
                tresh_img = layer.data
            elif len(layer.data.shape) > 4 or len(layer.data.shape) < 2:
                raise ValueError(
                    f"Image must be 2 or 3 dimensional. Not {len(layer.data.shape)} dimensions."
                )
        except IndexError:
            tresh_img = layer.data

        thresh_fig, best_method = create_threshold_hist_plot(tresh_img, file)
        save_plt(file, f"{layer.name}_thresholds", thresh_fig, "thresholds")

    return best_method


def run_tresh_and_segmentation(viewer: _layers, best_tresh: str):
    layers = {}
    for layer in viewer:
        tresh_layer = layer.data > THRESH_METHODS[best_tresh](layer.data)
        print("created tresh layer", layer.name)
        segmetation_layer = voronoi_otsu_labeling(
            tresh_layer, spot_sigma=2, outline_sigma=2
        )
        print("created segmetation layer", layer.name)
        binary = segmetation_layer > 0
        print("created binary layer", layer.name)

        layers[f"{layer.name}_{best_tresh}_threshold".lower()] = tresh_layer
        layers[f"{layer.name}_labels"] = segmetation_layer
        layers[f"{layer.name}_binary"] = binary

    for name, data in layers.items():
        viewer.add_layer(data, name=name)
        print(f"added {name} to viewer")


def save_layers(viewer: _layers, file: str) -> None:
    # save the layers in the viewer
    file_name = os.path.basename(file)
    # create a folder for layers
    os.makedirs(
        os.path.join(os.path.dirname(file), file_name.split(".")[0], "layers"),
        exist_ok=True,
    )
    for layer in viewer:
        try:
            imsave(
                os.path.join(
                    os.path.dirname(file),
                    file_name.split(".")[0],
                    "layers",
                    f"{layer.name}.tiff",
                ),
                np.asarray(layer.data),
            )
        except Exception as e:
            try:
                image = img_as_ubyte(layer.data)
                imsave(
                    os.path.join(
                        os.path.dirname(file),
                        file_name.split(".")[0],
                        "layers",
                        f"{layer.name}.tiff",
                    ),
                    image,
                )
            except Exception as e:
                FAILED[file]["save_layers"] = e


def calculate_colo(
    layer1: np.ndarray, binary: np.ndarray, properties: Tuple[str] = None
) -> np.ndarray:
    """Get an array with the same size as layer1 containing only the labels in layer1 colocalized with the binary array.

    Parameters
    ----------
    layer1 : np.ndarray
        The layer containg labels to be filtered for colocalization with binary
    binary : np.ndarray
        The layer containg labels to find colocalization against for layer1 to be filtered

    Returns
    -------
    np.ndarray
        The filtered layer1 array
    """
    if properties is None:
        properties = (
            "label",
            "mean_intensity",
            "area",
            "bbox",
            "centroid",
            "coords",
        )
    props = regionprops_table(layer1, binary, properties=properties)
    df = pd.DataFrame(props)
    colo = df[df["mean_intensity"] > 0]

    # Part 1. np.isin(layer1, colo["label"]) returns a boolean array with the same size as layer1.
    #     - If the element in layer1 is also in the 'colocalization with binary' array (colo["label"]),
    #       the corresponding element in the boolean array is True. If the element in layer1 is not in
    #       colo["label"], the corresponding element in the boolean array is False.
    # Part 2. np.where(np.isin(layer1, colo["label"]), layer1, 0)
    #     - If the element in the previously calculated boolean (np.isin) array is True, the returned
    #       element in the array is the corresponding label from the layer1 array. If there is no
    #       corresponding label in layer1, the returned element is 0.

    return np.where(np.isin(layer1, colo["label"]), layer1, 0)


def calculate_colocalization(viewer: Viewer, metadata: dict, file: str):
    for layer in viewer:
        # get the VGaT layer
        if layer.name == "VGaT_labels":
            vgat_labels = layer.data
        if layer.name == "VGaT_binary":
            vgat_binary = layer.data
        # get the GFP layer binary
        elif layer.name == "EGFP_binary":
            gfp_binary = layer.data
        # get the VGluT3
        elif layer.name == "VGluT3_labels":
            vglut3_labels = layer.data
        if layer.name == "VGluT3_binary":
            vglut3_binary = layer.data
    assert (
        vgat_labels is not None and gfp_binary is not None and vglut3_labels is not None
    ), "VGaT, GFP and VGluT3 layers are required for colocalization calculation."

    properties = (
            "label",
            "mean_intensity",
            "area",
            "bbox",
            "centroid",
            "coords",
    )
    # calculate colocalization for VGaT with GFP
    colo_vgat_gfp = calculate_colo(vgat_labels, gfp_binary, properties)
    # calculate colocalization for VGluT3 with GFP
    colo_vglut_gfp = calculate_colo(vglut3_labels, gfp_binary, properties)

    ####################################################################################
    ####### Henceforth we will only use the colocalization with GFP for the rest #######
    ####################### of the colocalization calculations #########################
    ####################################################################################

    # calculate colocalization for VGaT GFP+ with VGluT3
    colo_vgat_vglut = calculate_colo(colo_vgat_gfp, vglut3_binary, properties)

    # calculate colocalization for VGluT3 GFP+ with VGaT
    colo_vglut_vgat = calculate_colo(colo_vglut_gfp, vgat_binary, properties)

    imsave(
        os.path.join(
            os.path.dirname(file),
            os.path.basename(file).split(".")[0],
            "layers",
            "VGaT_colocalization.tiff",
        ),
        np.asarray(colo_vgat_vglut),
    )

    imsave(
        os.path.join(
            os.path.dirname(file),
            os.path.basename(file).split(".")[0],
            "layers",
            "VGluT3_colocalization.tiff",
        ),
        np.asarray(colo_vglut_vgat),
    )

    colo_vgat_vglut_table = pd.DataFrame(
        regionprops_table(colo_vgat_vglut, colo_vglut_vgat, properties=properties)
    )
    colo_vglut_vgat_table = pd.DataFrame(
        regionprops_table(colo_vglut_vgat, colo_vgat_vglut, properties=properties)
    )

    # convert the area to microns
    colo_vgat_vglut_table["area"] = (
        colo_vgat_vglut_table["area"] * metadata["pixel_microns"]
    )
    colo_vglut_vgat_table["area"] = (
        colo_vglut_vgat_table["area"] * metadata["pixel_microns"]
    )

    # rename the area column to colocalization area
    colo_vgat_vglut_table.rename(columns={"area": "area µm^2"}, inplace=True)
    colo_vglut_vgat_table.rename(columns={"area": "area µm^2"}, inplace=True)

    # save the colocalization tables
    file_name = os.path.basename(file)
    colo_vgat_vglut_table.to_csv(
        os.path.join(
            os.path.dirname(file),
            file_name.split(".")[0],
            f"{file_name.split('.')[0]}_VGaT_colocalization.csv",
        ),
        index=False,
    )
    colo_vglut_vgat_table.to_csv(
        os.path.join(
            os.path.dirname(file),
            file_name.split(".")[0],
            f"{file_name.split('.')[0]}_VGluT3_colocalization.csv",
        ),
        index=False,
    )

    # create a plot for number of VGaT and VGluT3
    fig, ax = plt.subplots()
    ax.bar(
        ["VGaT", "VGluT3"],
        [len(colo_vgat_vglut_table), len(colo_vglut_vgat_table)],
    )
    ax.set_title("Number of VGaT and VGluT3 colocalization in GFP+")
    ax.set_ylabel("Amount")
    ax.set_xlabel("VGaT and VGluT3")
    plt.savefig(
        os.path.join(
            os.path.dirname(file),
            file_name.split(".")[0],
            f"{file_name.split('.')[0]}colocalization_amount.png",
        )
    )

    # create a plot for the average size of colocalized VGaT and VGluT3
    fig, ax = plt.subplots()
    ax.bar(
        ["VGaT", "VGluT3"],
        [
            np.mean([prop.area for prop in regionprops(colo_vgat_gfp)]),
            np.mean([prop.area for prop in regionprops(colo_vglut_gfp)]),
        ],
    )
    ax.set_title("Average size of VGaT and VGluT3 colocalization in GFP+")
    ax.set_ylabel("Average size (µm^2)")
    ax.set_xlabel("VGaT and VGluT3")
    plt.savefig(
        os.path.join(
            os.path.dirname(file),
            file_name.split(".")[0],
            f"{file_name.split('.')[0]}colocalization_size.png",
        )
    )


cant_process = []


def runner(file: str):

    with ND2Reader(file) as images:
        metadata = images.metadata
        viewer = _layers()

        for i, channel in enumerate(metadata["channels"]):
            try:
                data = np.zeros(
                    (images.sizes["z"], images.sizes["x"], images.sizes["y"])
                )
                for t in range(images.sizes["z"]):
                    data[t, :, :] = images.get_frame_2D(c=i, z=t)

                viewer.add_layer(data, name=channel)

            except KeyError as e:
                cant_process.append(file)
                return

    for layer in viewer:
        if layer.name == "TRITC":
            layer.name = "VGaT"
        elif layer.name == "EGFP":
            layer.name = "EGFP"
        elif layer.name == "Cy5":
            layer.name = "VGluT3"

    best_tresh = create_threshold_plot(viewer, file)

    run_tresh_and_segmentation(viewer, best_tresh)

    # save all layers in the viewer
    save_layers(viewer, file)

    # calculate the colocalization
    calculate_colocalization(viewer, metadata, file)


def main(dir: str):
    warnings.filterwarnings("ignore")
    files = []
    if os.path.isfile(dir):
        files.append(dir)
    else:
        for file in os.listdir(dir):
            if file.endswith(".nd2"):
                file = os.path.join(dir, file)
                files.append(file)

    # create an instance of the pool class
    pool = Pool(cpu_count() - 1)
    # run the runner function on all files
    pool.map(runner, files)
    # start the pool
    pool.close()
    pool.join()
    # close the pool
    pool.terminate()
    if len(FAILED) != 0:
        for file, e in FAILED.items():
            print(f"failed to process {file}: {e}")
    else:
        print("all files processed successfully")


if __name__ == "__main__":
    input_dir = input("Enter the directory OR the file to be analyzed: ")
    main(input_dir)
