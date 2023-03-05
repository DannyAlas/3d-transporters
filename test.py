import napari
from napari_plugin_engine import napari_hook_implementation
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import local_maxima
from scipy import ndimage as ndi
from napari.types import ImageData, LabelsData
from skimage.io import imread

def voronoi_otsu_labeling(image:"napari.types.ImageData", spot_sigma: float = 2, outline_sigma: float = 2) -> "napari.types.LabelsData":

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
    labels = watershed(binary_otsu, labeled_spots, mask=binary_otsu)

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


# initialize viewer
viewer = napari.Viewer()


# add GFP images
GFP = imread(r"C:\Users\Daniel\Desktop\Work\lab\Data\drive-download-20230301T232528Z-002\R213\GFP\\*.tif")
viewer.add_image(
    GFP,
    colormap='green',
    blending='additive'
)

# add VGluT3 images
VGluT3 = imread(r"C:\Users\Daniel\Desktop\Work\lab\Data\drive-download-20230301T232528Z-002\R213\VGLUT3\\*.tif")
viewer.add_image(
    VGluT3,
    colormap='gray',
    blending='additive'
)

# add VGaT images
VGaT = imread(r"C:\Users\Daniel\Desktop\Work\lab\Data\drive-download-20230301T232528Z-002\R213\VGAT\\*.tif")
viewer.add_image(
    VGaT, 
    name='VGaT',
    colormap='red',
    blending='additive'
)


# add labels

# use gauss_otsu_labeling to get VGluT3 labels
VGluT3_lables = gauss_otsu_labeling(VGluT3)

VGluT3_lables_viewer = viewer.add_labels(
    VGluT3_lables,
)











input("Press Enter to continue...")
