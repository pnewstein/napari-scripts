"""
Steps for analysing images
"""

from __future__ import annotations
from typing import Protocol, Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray
from napari import Viewer
from napari.layers import Layer, Image, Labels, Points
from napari.types import ImageData, LabelsData, PointsData
from skimage import restoration, morphology, segmentation, measure, feature
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
    ) -> Labels | Image | Points: ...


def _make_analysis_step(
    function: Callable[..., ImageData | LabelsData],
    name_prefix: str,
    out_type: Literal["Image", "Labels", "Points"] = "Image",
    additive=False,
    show=True,
    distance_params: dict[str, bool] | None = None,
) -> AnalysisStep:
    """
    takes a function that takes in image data and returns an AnalysisStep
    distance params are all of the params related to distance and whether to round them after being scaled
    """
    if distance_params is None:
        distance_params = {}

    def out_function(
        viewer: Viewer,
        layer: int | Layer,
        name: str | None = None,
        *args,
        **kwargs,
    ) -> Labels | Image | Points:
        # remove kwargs that should not be passed to function
        if "show" in kwargs:
            del kwargs["show"]
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
                if distance_params[distance_param]:
                    kwargs[distance_param] = tuple(
                        (unscaled_value / layer.scale).round().astype(int).tolist()
                    )
                else:
                    kwargs[distance_param] = tuple(
                        (unscaled_value / layer.scale).tolist()
                    )
            elif scaled_value:
                # round scaled value if neccisary
                if distance_params[distance_param]:
                    scaled_value = int(round(scaled_value))
                kwargs[distance_param] = scaled_value
            if f"pix_{distance_param}" in kwargs:
                del kwargs[f"pix_{distance_param}"]
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
        elif out_type == "Points":
            new_layer = Points(
                out_data, name=name, translate=layer.translate, scale=layer.scale
            )
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
    # clear top and bottom
    mask = mask.copy()
    mask[[0, -1], :, :] = 0
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
    lbl: int, input_lbls: np.ndarray, dim_pix_mask: np.ndarray, thresh: float
) -> np.ndarray:
    """
    recursivly called function that merges a cell with its largest border neighbor
    that has the majority of pixels less than threshold. It then recalculates that
    merging potential with the newly merged cell
    thresh is the percentage of pixels that are in dim mask for a merge to occur
    """
    lbl_mask = input_lbls == lbl
    if not np.any(lbl_mask):
        return input_lbls
    expanded = ndi.binary_dilation(lbl_mask)
    edge = expanded & np.logical_not(lbl_mask)
    frac_dim_edge = (edge & dim_pix_mask).sum() / edge.sum()
    if frac_dim_edge == 0:
        return input_lbls
    sorted_neighbors = pd.Series(input_lbls[edge]).value_counts()
    if len(sorted_neighbors) == 1:
        # there is only one neighbor, so merge it in
        neighbor = sorted_neighbors.index[0]
        if neighbor != 0:
            print(lbl, neighbor)
            input_lbls[lbl_mask] = neighbor
            return input_lbls
    for neighbor in sorted_neighbors.index:
        cell_cell_border_mask = (input_lbls == neighbor) & edge
        if dim_pix_mask[cell_cell_border_mask].mean() > thresh:
            # merge the pixels
            input_lbls[lbl_mask] = neighbor
            assert lbl not in input_lbls
            if neighbor == 0:
                return input_lbls
            return maybe_merge_with_neighbors(
                neighbor, input_lbls, dim_pix_mask, thresh
            )
    # iterated through all lables
    return input_lbls


def _merge_dim_edged_labels(
    input_lbls: np.ndarray,
    mask: np.ndarray,
    thresh: float,
):

    input_lbls, _, _ = segmentation.relabel_sequential(input_lbls)
    # its important for this to be in order
    n_cells = input_lbls.max()
    for lbl in range(1, n_cells + 1):
        input_lbls = maybe_merge_with_neighbors(lbl, input_lbls, mask, thresh)
        print(lbl / n_cells)
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


def _blur(input_image: np.ndarray, sigma: tuple[float, float, float]) -> ImageData:
    """
    gauisan blur on image rescaled to fit in in the dtype
    """

    scale_max = np.iinfo(input_image.dtype).max
    out_image = input_image * int(0.9 * scale_max / input_image.max())
    return cast(ImageData, ndi.gaussian_filter(out_image, sigma))


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
        _blur,  # type: ignore
        name_prefix="blur",
        distance_params={"sigma": False},
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
        distance_params={"size": False},
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
        distance_params={"outline_sigma": False, "spot_sigma": False},
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
        out_type="Image",
        distance_params={"size": True},
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
        distance_params={"size": False},
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
        distance_params={"size": False},
    )(viewer=viewer, layer=layer, name=name, size=size, pix_size=pix_size, show=show)
    if isinstance(out, Image):
        raise ValueError()
    return out



def local_maxima_to_points(
    viewer: Viewer,
    layer: int | Layer,
    threshold_abs: int,
    min_distance=1,
    name: str | None = None,
    show=True,
) -> Points:
    """
    converts local maxima to points
    """
    out = _make_analysis_step(
        feature.peak_local_max,
        name_prefix="maxima",
        out_type="Points",
        distance_params={"size": False},
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        show=show,
        threshold_abs=threshold_abs,
        min_distance=min_distance,
    )
    if isinstance(out, Image):
        raise ValueError()
    return out


def remove_labels_enclaves(labels: Labels):
    """
    remove all of the labels that are completely within another label
    """
    lbls = labels.data
    assert isinstance(lbls, np.ndarray)
    for lbl in np.unique(lbls):
        lbl_mask = lbls == lbl
        dialated_mask = cast(NDArray[np.bool_], ndi.binary_dilation(lbl_mask))
        lbl_and_neighbors = np.unique(lbls[dialated_mask]).tolist()
        if len(lbl_and_neighbors) != 2:
            continue
        lbl_and_neighbors.pop(lbl_and_neighbors.index(lbl))
        (other_lbl,) = lbl_and_neighbors
        lbls[lbl_mask] = other_lbl


def _local_minima_watershed(
    outline_blured: np.ndarray, minima_blured: np.ndarray, mask: NDArray | None = None
) -> LabelsData:
    """
    finds minima with minima_blured and uses that to seed a watershed from outline_blured
    """
    minimum_spots = measure.label(morphology.local_minima(minima_blured))
    return cast(
        LabelsData, segmentation.watershed(outline_blured, minimum_spots, mask=mask)
    )


def manual_thresh(
    viewer: Viewer,
    layer: int | Layer,
    thresh: float,
    name: str | None = None,
    show=True,
) -> Labels:
    out = _make_analysis_step(
        lambda layer, thresh: layer > thresh,
        name_prefix=f"threshold {thresh}",
        out_type="Labels",
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        thresh=thresh,
        show=show,
    )
    assert isinstance(out, Labels)
    return out


def get_npix_at_label(lbls: Labels) -> pd.Series:
    """
    returns the number of pixels of each label (excluing zero)
    """
    out = pd.Series(lbls.data.ravel()).value_counts()
    out.name = "Pixel count"
    return out.drop(0)


def quantify_points_in_labels(points: Points, lbls: Labels) -> pd.Series:
    """
    returns a series of how many points are in each label
    """
    out_series = pd.Series(0, name=points.name, index=np.unique(np.array(lbls.data)))
    for coords in points.data:
        pz, py, px = coords
        out_series.loc[lbls.data[pz, py, px]] += 1
    assert out_series.index[0] == 0
    return out_series.drop(0)


def local_minima_watershed(
    viewer: Viewer,
    layer: int | Layer,
    minima_blured: np.ndarray,
    name: str | None = None,
    show=True,
    mask: NDArray | None = None,
) -> Labels:
    """
    does a local minimum seaded watershed on the two images
    """
    out = _make_analysis_step(
        _local_minima_watershed,
        name_prefix="cells",
        out_type="Labels",
        distance_params={"outline_sigma": False, "spot_sigma": False},
    )(
        viewer=viewer,
        layer=layer,
        name=name,
        minima_blured=minima_blured,
        show=show,
        mask=mask,
    )
    assert isinstance(out, Labels)
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
        distance_params={"outline_sigma": False, "spot_sigma": False},
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
    viewer: Viewer,
    layer: int | Layer,
    name: str | None = None,
    show=True,
    top_bottom=True,
) -> Labels:
    """
    uses otsu thresholding on the image
    """
    out = _make_analysis_step(_clear_edges, name_prefix="no_edge", out_type="Labels")(
        viewer=viewer, layer=layer, name=name, show=show, top_bottom=top_bottom
    )
    if isinstance(out, Image):
        raise ValueError()
    return out


def merge_dim_edged_labels(
    viewer: Viewer,
    layer: int | Layer,
    mask: LabelsData,
    thresh: float = 0.5,
    name: str | None = None,
    show=True,
) -> Labels:
    out = _make_analysis_step(
        _merge_dim_edged_labels,
        name_prefix="bright_edges",
        out_type="Labels",
        distance_params={"sigma": False},
    )(viewer=viewer, layer=layer, name=name, show=show, mask=mask, thresh=thresh)
    if isinstance(out, Image):
        raise ValueError()
    return out
