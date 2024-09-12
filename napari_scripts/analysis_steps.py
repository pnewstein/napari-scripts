"""
Steps for analysing images
"""

from typing import Protocol, Callable, Literal, cast

import numpy as np
from napari import Viewer
from napari.layers import Layer, Image, Labels
from napari.types import ImageData, LabelsData
from skimage import restoration, morphology, filters
import scipy.ndimage as ndi
import pandas as pd
import napari_segment_blobs_and_things_with_membranes as nsbatwm

from napari_scripts.image_layers import img_layer


def _contrast(
    image: ImageData,
    contrast_min: float,
    contrast_max: float,
    knee: float | None = None,
) -> ImageData:
    """
    returns an array with the contrast burned in so that there are 3 linear segments
    knee is a float between 0 and 1 specifies the ratio of intensity at contrast_min vs max_intensity.
    constrast_min is the pixel intensity value where the output intensity begins to increase more rapidly
    returns image transormed to 8 bit
    """
    float_type = np.float32
    image_max = image.max()
    assert len(image_max.shape) == 0
    assert image_max > contrast_max > contrast_min
    image = cast(ImageData, image.astype(float_type))
    if knee is None:
        knee = 0.05
    below_knee = image < contrast_min
    above_knee = image > contrast_max
    between_knee = np.logical_not(np.logical_or(below_knee, above_knee))
    out_array = np.zeros(image.shape).astype(float_type)
    begining_slope = 255 * knee / contrast_min if contrast_min != 0 else 0
    np.multiply(float_type(begining_slope), image, out=out_array, where=below_knee)
    middle_slope = 255 * (1 - 2 * knee) / (contrast_max - contrast_min)
    middle_intercept = (
        255
        * (-contrast_min + knee * contrast_min + knee * contrast_max)
        / (contrast_max - contrast_min)
    )
    np.multiply(float_type(middle_slope), image, out=out_array, where=between_knee)
    np.add(float_type(middle_intercept), out_array, out=out_array, where=between_knee)
    ending_slope = 255 * knee / (image_max - contrast_max)
    ending_intercept = (
        255 * (image_max - contrast_max - knee * image_max) / (image_max - contrast_max)
    )
    np.multiply(float_type(ending_slope), image, out=out_array, where=above_knee)
    np.add(float_type(ending_intercept), out_array, out=out_array, where=above_knee)
    out = cast(ImageData, out_array.astype(np.uint16))
    return out


class AnalysisStep(Protocol):
    """
    A step in analysis which returns the image data and adds it to the
    """

    def __call__(
        self,
        viewer: Viewer,
        layer: int | Layer,
        name: str | None = None,
        *args,
        **kwargs,
    ) -> Labels | Image: ...


def _make_analysis_step(
    function: Callable[..., ImageData | LabelsData],
    name_prefix: str,
    out_type: Literal["Image", "Labels"] = "Image",
    additive=False,
    show=True,
    distance_params: list[str] | None = None,
) -> AnalysisStep:
    """
    takes a function that takes in image data and returns an AnalysisStep
    """
    if distance_params is None:
        distance_params = []

    def out_function(
        viewer: Viewer,
        layer: int | Layer,
        name: str | None = None,
        *args,
        **kwargs,
    ) -> Labels | Image:
        # set_default_layer
        if isinstance(layer, Layer):
            pass
        elif layer >= 0:
            layer = img_layer(viewer, layer)
        else:
            layer = viewer.layers[layer]
        # set defualt name
        if name is None:
            if "-" in layer.name:
                name = "-".join([name_prefix] + layer.name.split("-")[1:])
            else:
                name = function.__name__
        # rescale scalable params
        for distance_param in distance_params:
            unscaled_value = kwargs.get(distance_param)
            scaled_value = kwargs.get(f"pix_{distance_param}")
            if unscaled_value:
                kwargs[distance_param] = tuple((unscaled_value / layer.scale).tolist())
            elif scaled_value:
                kwargs[distance_param] = scaled_value
        # call function
        out_data = function(layer.data, *args, **kwargs)
        if np.issubdtype(out_data.dtype, np.floating):
            min_val = np.min(out_data)
            max_val = np.max(out_data)
            scaled_arr = (
                (out_data - min_val) / (max_val - min_val) * np.iinfo(np.uint16).max
            )
            out_data = scaled_arr.astype(np.uint16)
        if out_data.dtype == np.int8 and out_type == "Labels":
            assert out_data.min() == 0
            out_data = out_data.astype(np.uint8)
        if out_type == "Image":
            new_layer = Image(
                out_data, name=name, translate=layer.translate, scale=layer.scale
            )
            if isinstance(new_layer, list):
                raise ValueError()
            if additive:
                new_layer.blending = "additive"
        elif out_type == "Labels":
            new_layer = Labels(
                out_data, name=name, translate=layer.translate, scale=layer.scale
            )
            new_layer.contour = 1
        if show:
            viewer.layers.append(new_layer)
        return new_layer

    return out_function


def print_contrast_limits(viewer: Viewer):
    """
    prints out the contrast limits of the last image in a nice way
    """
    layer = viewer.layers[-1]
    if not isinstance(layer, Image):
        print("last layser is  not an image")
        return
    clims = layer.contrast_limits
    if clims:
        contrast_min, contrast_max = clims
        print(f"{(contrast_min, contrast_max)=}")


def ellipsoid_dialation_erosion(
    mask: np.ndarray, size: tuple[int, ...], direction: Literal["erosion", "dialation"]
) -> LabelsData:
    """
    npix is the 3 dimentailnal shape
    """
    out = np.zeros(mask.shape, dtype=mask.dtype)
    kernel = restoration.ellipsoid_kernel(size, 1) != np.inf
    for lbl in np.unique(mask):
        if direction == "dialation":
            this_mask = morphology.binary_dilation(mask == lbl, kernel)
        elif direction == "erosion":
            this_mask = morphology.binary_erosion(mask == lbl, kernel)
        out += this_mask * lbl
    return cast(LabelsData, out)


def maybe_merge_with_neighbors(
    lbl: int, input_lbls: np.ndarray, dim_pix_mask: np.ndarray
) -> np.ndarray:
    """
    recursivly called function that merges a cell with its largest border neighbor
    that has the majority of pixels less than threshold. It then recalculates that
    merging potential with the newly merged cell
    """
    lbl_mask = input_lbls == lbl
    if lbl_mask.sum() == 0:
        return input_lbls
    expanded = ndi.binary_dilation(lbl_mask)
    edge = expanded & np.logical_not(lbl_mask)
    frac_dim_edge = (edge & dim_pix_mask).sum() / edge.sum()
    if frac_dim_edge == 0:
        return input_lbls
    sorted_neighbors = pd.Series(input_lbls[edge]).value_counts()
    for neighbor in sorted_neighbors.index:
        cell_cell_border_mask = (input_lbls == neighbor) & edge
        if dim_pix_mask[cell_cell_border_mask].mean() > 0.5:
            # merge the pixels
            input_lbls[lbl_mask] = neighbor
            assert lbl not in input_lbls
            if neighbor == 0:
                return input_lbls
            return maybe_merge_with_neighbors(neighbor, input_lbls, dim_pix_mask)
    # iterated through all lables
    return input_lbls


def _merge_dim_edged_labels(
    input_lbls: np.ndarray, image: np.ndarray, sigma: tuple[float, ...]
):

    input_lbls = input_lbls.copy()
    blured = nsbatwm.gaussian(image, sigma)
    thresh = filters.threshold_minimum(blured)
    dim_pix_mask = blured < thresh
    # its important for this to be in order
    for lbl in range(1, input_lbls.max() + 1):
        input_lbls = maybe_merge_with_neighbors(lbl, input_lbls, dim_pix_mask)
    return cast(LabelsData, input_lbls)


def _clear_edges(mask: np.ndarray, top_bottom=True):
    """
    sets all labels on borders to 0
    """
    print(top_bottom)
    assert len(mask.shape) == 3
    borders = np.zeros(shape=mask.shape, dtype=np.uint8)
    if top_bottom:
        borders[[0, -1], :, :] = 1
    borders[:, [0, -1], :] = 1
    borders[:, :, [0, -1]] = 1
    lbls_on_edge = np.unique(mask[borders.astype(bool)])
    out = mask.copy()
    assert out is not mask
    out[np.isin(mask, lbls_on_edge)] = 0
    return cast(LabelsData, out)


def blur(
    viewer: Viewer,
    layer: int | Layer,
    show=True,
    name: str | None = None,
    sigma: float | None = None,
    pix_sigma: float | None = None,
) -> Image:
    """
    performs a gauisan blur on layer. sigma is the std deviation of the gauisan blur
    """
    return _make_analysis_step(
        ndi.gaussian_filter,  # type: ignore
        name_prefix="blur",
        distance_params=["sigma"],
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        sigma=sigma,
        pix_sigma=pix_sigma,
        show=show,
    )


def median(
    viewer: Viewer,
    layer: int | Layer,
    name: str | None = None,
    show=True,
    size: float | None = None,
    pix_size: int | None = None,
) -> Image:
    """
    performs a median blur on layer. size is the window size in pixels
    """

    def inner_filter(**kwargs) -> Image:
        if kwargs["size"]:
            kwargs["size"] = int(kwargs["size"])
        return ndi.median_filter(**kwargs)

    return _make_analysis_step(
        inner_filter,  # type: ignore
        name_prefix="median",
        distance_params=["size"],
    )(viewer=viewer, layer=layer, name=name, size=size, pix_size=pix_size, show=show)


def contrast(
    viewer: Viewer,
    layer: int | Layer,
    contrast_min: float,
    contrast_max: float,
    knee: float | None = None,
    name: str | None = None,
    show=True,
) -> Image:
    """
    returns an array with the contrast burned in so that there are 3 linear
    segments knee is a float between 0 and 1 specifies the ratio of intensity
    at contrast_min vs max_intensity. constrast_min is the pixel intensity
    value where the output intensity begins to increase more rapidly returns
    image transormed to 8 bit
    """
    out = _make_analysis_step(
        _contrast,
        name_prefix="contrast",
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        contrast_min=contrast_min,
        contrast_max=contrast_max,
        knee=knee,
        show=show,
    )
    if isinstance(out, Labels):
        raise ValueError()
    return out


def find_blobs(
    viewer: Viewer,
    layer: int | Layer,
    name: str | None = None,
    show=True,
    spot_sigma: float | None = None,
    outline_sigma: float | None = None,
    pix_spot_sigma: float | None = None,
    pix_outline_sigma: float | None = None,
) -> Labels:
    """
    Voronoi-Otsu-Labeling is a segmentation algorithm for blob-like structures
    such as nuclei and granules with high signal intensity on low-intensity
    background.
    The two sigma parameters allow tuning the segmentation result. The first
    sigma controls how close detected cells can be (spot_sigma) and the second
    controls how precise segmented objects are outlined (outline_sigma). Under
    the hood, this filter applies two Gaussian blurs, spot detection,
    Otsu-thresholding and Voronoi-labeling.
    """
    out = _make_analysis_step(
        nsbatwm.voronoi_otsu_labeling,
        name_prefix="blobs",
        out_type="Labels",
        distance_params=["outline_sigma", "spot_sigma"],
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        pix_spot_sigma=pix_spot_sigma,
        pix_outline_sigma=pix_outline_sigma,
        show=show,
    )
    if isinstance(out, Image):
        raise ValueError()
    return out


def threshold(
    viewer: Viewer, layer: int | Layer, name: str | None = None, show=True
) -> Labels:
    """
    uses otsu thresholding on the image
    """
    out = _make_analysis_step(
        nsbatwm.threshold_otsu, name_prefix="otsu", out_type="Labels"
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        show=show,
    )
    if isinstance(out, Image):
        raise ValueError()
    return out


def tophat(
    viewer: Viewer,
    layer: int | Layer,
    name: str | None = None,
    size: float | None = None,
    pix_size: float | None = None,
    show=True,
) -> Image:
    """
    removes things biger than size
    """
    out = _make_analysis_step(
        ndi.white_tophat,  # type: ignore
        name_prefix="tophat",
        out_type="Labels",
        distance_params=["size"],
    )(viewer=viewer, layer=layer, name=name, size=size, pix_size=pix_size, show=show)
    if isinstance(out, Labels):
        raise ValueError()
    return out


def binary_dilation(
    viewer: Viewer,
    layer: int | Layer,
    name: str | None = None,
    size: float | None = None,
    pix_size: float | None = None,
    show=True,
) -> Labels:
    """
    removes things biger than size
    """
    out = _make_analysis_step(
        lambda mask, size: ellipsoid_dialation_erosion(mask, size, "dialation"),
        name_prefix="dialation",
        out_type="Labels",
        distance_params=["size"],
    )(viewer=viewer, layer=layer, name=name, size=size, pix_size=pix_size, show=show)
    if isinstance(out, Image):
        raise ValueError()
    return out


def binary_erosion(
    viewer: Viewer,
    layer: int | Layer,
    name: str | None = None,
    size: float | None = None,
    pix_size: float | None = None,
    show=True,
) -> Labels:
    """
    removes things biger than size
    """
    out = _make_analysis_step(
        lambda mask, size: ellipsoid_dialation_erosion(mask, size, "erosion"),
        name_prefix="erosion",
        out_type="Labels",
        distance_params=["size"],
    )(viewer=viewer, layer=layer, name=name, size=size, pix_size=pix_size, show=show)
    if isinstance(out, Image):
        raise ValueError()
    return out


def within_membranes(
    viewer: Viewer,
    layer: int | Layer,
    name: str | None = None,
    show=True,
    spot_sigma: float | None = None,
    outline_sigma: float | None = None,
    pix_spot_sigma: float | None = None,
    pix_outline_sigma: float | None = None,
) -> Labels:
    """
    Segment cells in images with fluorescently marked membranes.

    The two sigma parameters allow tuning the segmentation result. The first
    sigma controls how close detected cells can be (spot_sigma) and the second
    controls how precise segmented objects are outlined (outline_sigma). Under
    the hood, this filter applies two Gaussian blurs, local minima detection
    and a seeded watershed.
    """
    out = _make_analysis_step(
        nsbatwm.local_minima_seeded_watershed,
        name_prefix="cells",
        out_type="Labels",
        distance_params=["outline_sigma", "spot_sigma"],
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        pix_spot_sigma=pix_spot_sigma,
        pix_outline_sigma=pix_outline_sigma,
        show=show,
    )
    if isinstance(out, Image):
        raise ValueError()
    return out


def clear_edges(
    viewer: Viewer, layer: int | Layer, name: str | None = None, show=True
) -> Labels:
    """
    uses otsu thresholding on the image
    """
    out = _make_analysis_step(_clear_edges, name_prefix="no_edge", out_type="Labels")(
        viewer=viewer,
        layer=layer,
        name=name,
        show=show,
    )
    if isinstance(out, Image):
        raise ValueError()
    return out


def merge_dim_edged_labels(
    viewer: Viewer,
    layer: int | Layer,
    image: ImageData,
    sigma: float | None = None,
    pix_sigma: float | None = None,
    name: str | None = None,
    show=True,
) -> Labels:
    out = _make_analysis_step(
        _merge_dim_edged_labels,
        name_prefix="bright_edges",
        out_type="Labels",
        distance_params=["sigma"],
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        show=show,
        sigma=sigma,
        pix_sigma=pix_sigma,
        image=image
    )
    if isinstance(out, Image):
        raise ValueError()
    return out
