from asyncio.log import logger
from importlib.metadata import metadata
import os
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
from napari import Viewer
import numpy as np
from napari.types import ImageData, LabelsData
from skimage.filters import threshold_isodata, threshold_li, threshold_mean, threshold_minimum, threshold_otsu, threshold_triangle, threshold_yen, gaussian
from skimage.morphology import local_maxima, remove_small_objects, dilation
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import relabel_sequential as _relabel_sequential, clear_border as _clear_border, expand_labels, watershed as _watershed, random_walker as _random_walker
from skimage.exposure import rescale_intensity
import pandas as pd
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.handlers = []


def voronoi_otsu_labeling(image:ImageData, spot_sigma: float = 2, outline_sigma: float = 2) -> LabelsData:
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

def gauss_otsu_labeling(image:ImageData, outline_sigma: float = 2) -> LabelsData:
    """Gauss-Otsu-Labeling can be used to segment objects such as nuclei with bright intensity on
    low intensity background images.

    Parameters
    ----------
    image : ndarray
        Input image.
    outline_sigma : float, optional
        Standard deviation of the Gaussian kernel used to smooth the outlines. Controls how precise segmented objects are.
    
    Returns
    -------
    labels : ndarray
        Labeled image.
    """
    image = np.asarray(image)

    # blur
    blurred_outline = gaussian(image, outline_sigma)

    # threshold
    threshold = threshold_otsu(blurred_outline)
    binary_otsu = blurred_outline > threshold

    # connected component labeling
    labels = label(binary_otsu)

    return labels

def set_napari_channels(viewer: Viewer, metadata):
    for i, channel in enumerate(metadata['channels']):
        logger.info(f'Channel: {channel} ({i})')
        # rename TRITC to VGaT
        if channel == 'TRITC':
            channel = 'VGaT'
        # rename CY5 to VGLUT3
        if channel == 'Cy5':
            channel = 'VGluT3'
        # create a variable for each channel
        exec(f'{channel} = viewer.layers[{i}].data')
        logger.debug(f'Channel: {channel} ({i}) - {viewer.layers[i].data.shape} - {viewer.layers[i].data.dtype} - {viewer.layers[i].data.nbytes} bytes')
        # rename the layer
        viewer.layers[i].name = channel


    # set the color of layers
    for i, layer in enumerate(viewer.layers):
        if layer.name == 'DAPI':
            layer.colormap = 'blue'
        elif layer.name == 'EGFP':
            layer.colormap = 'green'
        elif layer.name == 'VGaT':
            layer.colormap = 'red'
        elif layer.name == 'VGluT3':
            layer.colormap = 'magenta'

def save_plt(file, name, plt: plt, folder: str = None):
    # get the file name
    file_name = os.path.basename(file)
    # check if a folder exists for the file
    if not os.path.exists(os.path.join(os.path.dirname(file), file_name.split('.')[0])):
        os.mkdir(os.path.join(os.path.dirname(file), file_name.split('.')[0]))
    # check if a folder is given
    if folder:
        logger.debug(f'Folder: {folder}')
        # check if the folder exists
        if not os.path.exists(os.path.join(os.path.dirname(file), file_name.split('.')[0], folder)):
            os.mkdir(os.path.join(os.path.dirname(file), file_name.split('.')[0], folder))
            logger.debug(f'Folder created: {os.path.join(os.path.dirname(file), file_name.split(".")[0], folder)}')
        # save the plot
        plt.savefig(os.path.join(os.path.dirname(file), file_name.split('.')[0], folder, name + '.png'))
        logger.info(f'Plot saved: {os.path.join(os.path.dirname(file), file_name.split(".")[0], folder, name + ".png")}')
    else:
        # save the plot
        plt.savefig(os.path.join(os.path.dirname(file), file_name.split('.')[0], name + '.png'))
        logger.info(f'Plot saved: {os.path.join(os.path.dirname(file), file_name.split(".")[0], name + ".png")}')

def increase_contrast(image: ImageData, low: float=0.5, high: float=99.5) -> ImageData:
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

def create_threshold_hist_plot(image: np.ndarray):
    """Create a plot with a histogram of the image and the threshold of different thresholding methods.
    """
    methods = {'Isodata': threshold_isodata,'Li': threshold_li,'Mean': threshold_mean,'Minimum': threshold_minimum,'Otsu': threshold_otsu,'Triangle': threshold_triangle,'Yen': threshold_yen}
    
    logger.info(f'Creating threshold histogram plot')
    logger.debug(f'Image shape: {image.shape}')

    if len(image.shape) > 2:
        raise ValueError('Image must be 2 dimensional.')
    
    fig, axes = plt.subplots(2, len(methods), figsize=(20, 5))
    for i, (method, func) in enumerate(methods.items()):
        axes[0, i].imshow(image > func(image), cmap='gray')
        axes[0, i].set_title(method)
        axes[0, i].axis('off')
        axes[1, i].hist(image.ravel(), bins=256)
        thresh = func(image)
        axes[1, i].axvline(thresh, color='r')
        axes[1, i].set_title(f'{method} threshold: {int(thresh)}')
    
    
    fig.tight_layout(h_pad=0.2, w_pad=0.1)

    return fig

def create_threshold_plot(viewer: Viewer, file: str):
    for layer in viewer.layers:
        try:
            if len(layer.data.shape) ==3:
                tresh_img = layer.data[0]
            elif len(layer.data.shape) == 2:
                tresh_img = layer.data
            elif len(layer.data.shape) > 4 or len(layer.data.shape) < 2:
                raise ValueError(f'Image must be 2 or 3 dimensional. Not {len(layer.data.shape)} dimensions.')
        except IndexError:
            tresh_img = layer.data
        
        thresh_fig = create_threshold_hist_plot(tresh_img)
        save_plt(file, f'{layer.name}_thresholds', thresh_fig, 'thresholds')

def main():
    dir = r"C:\Users\Daniel\Desktop\Work\lab\Data\drive-download-20230301T232528Z-001"

    for file in os.listdir(dir):
        if file.endswith(".nd2"):
            file = os.path.join(dir, file)
            with ND2Reader(file) as images:
                metadata = images.metadata
            
            viewer = Viewer()
            viewer.open(file, plugin='napari-nikon-nd2')

            set_napari_channels(viewer, metadata)

            create_threshold_plot(viewer, file)
            



            viewer.close()



            

if __name__ == '__main__':
    main()