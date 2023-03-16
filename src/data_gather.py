import logging
import os
import warnings
from asyncio.log import logger
from copy import copy, deepcopy
from importlib.metadata import metadata
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from napari import Viewer
from napari.types import ImageData, LabelsData
from nd2reader import ND2Reader
from skimage.exposure import rescale_intensity
from skimage.filters import (
    gaussian,
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.io import imsave
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import local_maxima
from skimage.segmentation import clear_border as _clear_border
from skimage.segmentation import expand_labels
from skimage.segmentation import random_walker as _random_walker
from skimage.segmentation import relabel_sequential as _relabel_sequential
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
        FAILED[file] = e

    # determine the best thresholding method by the threshold that has the lowes misclassification rate
    misclassification_rate = {}
    for method, values in threshes.items():
        # FIXME: this is just temporary, a better way to determine the best thresholding method should be implemented
        misclassification_rate[method] = np.abs(values["above"] - values["below"])
        # best method is the method with the misclassification rate closest to 0
        # best_method = min(misclassification_rate, key=misclassification_rate.get)

    fig.tight_layout(h_pad=0.2, w_pad=0.1)

    return fig, "Otsu"


def create_threshold_plot(viewer: Viewer, file: str):
    for layer in viewer.layers:
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


def run_tresh_and_segmentation(viewer: Viewer, best_tresh: str):
    layers = {}
    for layer in viewer.layers:
        tresh_layer = layer.data > THRESH_METHODS[best_tresh](layer.data)
        print("created tresh layer")
        segmetation_layer = voronoi_otsu_labeling(
            tresh_layer, spot_sigma=2, outline_sigma=2
        )
        print("created segmetation layer")
        binary = segmetation_layer > 0
        props = regionprops_table(
            segmetation_layer, binary, properties=("label", "mean_intensity")
        )
        df = pd.DataFrame(props)
        colo = df[df["mean_intensity"] > 0.1]
        labels = np.where(
            np.isin(segmetation_layer, colo["label"]), segmetation_layer, 0
        )
        print("created colo_labels layer")
        layers[f"{layer.name}_segmentation"] = segmetation_layer
        layers[f"{layer.name}_{best_tresh}_threshold".lower()] = tresh_layer
        layers[f"{layer.name}_binary"] = binary
        layers[f"{layer.name}_labels"] = labels

    for name, data in layers.items():
        if name.endswith("segmentation"):
            viewer.add_labels(data, name=name)
        elif name.endswith("labels"):
            viewer.add_labels(data, name=name)
        else:
            viewer.add_image(data, name=name)
        print(f"added {name} to viewer")


def save_layers(viewer: Viewer, file: str):
    # save the layers in the viewer
    file_name = os.path.basename(file)
    # create a folder for layers
    os.makedirs(
        os.path.join(os.path.dirname(file), file_name.split(".")[0], "layers"),
        exist_ok=True,
    )
    for layer in viewer.layers:
        if layer.data is None:
            continue
        try:
            image = img_as_ubyte(layer.data)
            # save the image as a tiff
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
            FAILED[file] = e


def calculate_colocalization(viewer: Viewer, metadata: dict, file: str):
    for layer in viewer.layers:
        # get the VGaT layer
        if layer.name == "VGaT_labels":
            vgat_labels = layer.data
        # get the GFP layer binary
        elif layer.name == "EGFP_binary":
            gfp_binary = layer.data
        # get the VGluT3
        elif layer.name == "VGluT3_labels":
            vglut3_labels = layer.data

    properties = ("label", "mean_intensity", "area", "bbox")
    props = regionprops_table(vgat_labels, gfp_binary, properties=properties)
    df = pd.DataFrame(props)
    colo = df[df["mean_intensity"] > 0]
    colo_vgat_gfp = np.where(np.isin(vgat_labels, colo["label"]), vgat_labels, 0)

    props = regionprops_table(
        vglut3_labels,
        gfp_binary,
        properties=("label", "mean_intensity", "area", "bbox"),
    )
    df = pd.DataFrame(props)
    colo = df[df["mean_intensity"] > 0]
    colo_vglut_gfp = np.where(np.isin(vglut3_labels, colo["label"]), vglut3_labels, 0)

    colo_vgat_gfp_table = pd.DataFrame(
        regionprops_table(colo_vgat_gfp, properties=("label", "area", "bbox"))
    )
    colo_vglut_gfp_table = pd.DataFrame(
        regionprops_table(colo_vglut_gfp, properties=("label", "area", "bbox"))
    )

    # convert the area to microns
    colo_vgat_gfp_table["area"] = (
        colo_vgat_gfp_table["area"] * metadata["pixel_microns"]
    )
    colo_vglut_gfp_table["area"] = (
        colo_vglut_gfp_table["area"] * metadata["pixel_microns"]
    )

    # rename the area column to colocalization area
    colo_vgat_gfp_table.rename(columns={"area": "area µm^2"}, inplace=True)
    colo_vglut_gfp_table.rename(columns={"area": "area µm^2"}, inplace=True)

    # save the colocalization table
    file_name = os.path.basename(file)
    colo_vgat_gfp_table.to_csv(
        os.path.join(
            os.path.dirname(file), file_name.split(".")[0], "VGaT_colocalization.csv"
        ),
        index=False,
    )
    colo_vglut_gfp_table.to_csv(
        os.path.join(
            os.path.dirname(file), file_name.split(".")[0], "VGluT3_colocalization.csv"
        ),
        index=False,
    )


def runner(file: str):
    with ND2Reader(file) as images:
        metadata = images.metadata

    # start napaari viewer with no gui
    viewer = Viewer(show=False)
    try:
        viewer.open(file, plugin="napari-nikon-nd2")
    except Exception as e:
        FAILED[file] = e
        return
    set_napari_channels(viewer, metadata)

    best_tresh = create_threshold_plot(viewer, file)

    run_tresh_and_segmentation(viewer, best_tresh)

    # save all layers in the viewer
    save_layers(viewer, file)

    # calculate the colocalization
    calculate_colocalization(viewer, metadata, file)

    # close the viewer
    viewer.close()


def main():
    warnings.filterwarnings("ignore")
    dir = r"C:\Users\Daniel\Desktop\Work\lab\Data\drive-download-20230301T232528Z-001"
    files = []
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
    main()
