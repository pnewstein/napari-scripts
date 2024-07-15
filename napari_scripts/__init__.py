"""
Some scripts for managing napari
"""

from __future__ import annotations
from pathlib import Path
from typing import Generator, Iterable, Sequence, Callable, Protocol, Literal
from dataclasses import dataclass
import random
import json
import re
import os

import numpy as np
from napari_czifile2 import reader_function_with_args, SceneIndexOutOfRange
from napari_czifile2.io import CZISceneFile
from napari.layers import Image
from napari.viewer import Viewer
from napari.types import ImageData, LabelsData, LayerData
import napari
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import pandas as pd
from tifffile import TiffFile, TiffFrame

FLUOROPHORE_LIST_PATH = Path(__file__).parent / "fluorophore_list.json"


COLOR_MAP = {"g": "green", "m": "magenta", "k": "gray", "r": "red", "p": "PiYG", "c": "cyan", "y": "yellow"}

def get_fluorophore_list() -> list[str]:
    """
    gets fluorophore_list from disk
    """
    return json.loads(FLUOROPHORE_LIST_PATH.read_text("utf-8"))


def set_fluorophoe_list(fluorophore_list: list[str]):
    """
    sets fluorophore_list to disk
    """
    FLUOROPHORE_LIST_PATH.write_text(json.dumps(fluorophore_list), "utf-8")


@dataclass(frozen=True)
class PathScene:
    """
    A path and a scene to point to a czi file
    """

    path: Path
    scene: int

    def to_tuple(self) -> tuple[str, int]:
        """
        convert to a tuple for serialization
        """
        return (str(self.path), self.scene)

    @classmethod
    def from_tuple(cls, tup: Iterable):
        """
        deserializes the data expects a two value iterable
        """
        path_str, scene = tup
        return cls(path=Path(path_str), scene=scene)


def generate_random_key(key_path: Path, image_paths: Iterable[Path]):
    """
    creates a list of tuples of
    (file_path (str), scene_num (int))
    and saves it to key_path for use in
    get_random_viewer
    """
    # do work on the flyroom pc
    if key_path.exists():
        raise FileExistsError(f"refusing to overwrite {key_path}")
    path_scenes: list[PathScene] = []
    for path in image_paths:
        if key_path.suffix == ".czi":
            n_scenes = CZISceneFile.get_num_scenes(path)
        else:
            n_scenes = 1
        path_scenes.extend(PathScene(path, i) for i in range(n_scenes))
    random.shuffle(path_scenes)
    print(len(path_scenes))
    key_path.write_text(json.dumps([ps.to_tuple() for ps in path_scenes], indent=4))


def get_random_viewer(key_path: Path, img_num: int) -> Viewer:
    """
    gets the viewer at img_num from key_path
    """
    this_path_scene = PathScene.from_tuple(
        json.loads(key_path.read_text("utf-8"))[img_num]
    )
    return get_viewer_from_file(this_path_scene.path, this_path_scene.scene, display_num=img_num)



def get_viewer_from_file(image_path: Path, scene_num: int, display_num=None) -> Viewer:
    """
    gets a viewer with the file opened at scene. tries reading czis or tiffs
    """
    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.
    if image_path.parts[:3] == ('/', 'Volumes', 'DoeLab65TB') and os.name == "nt":
        image_path = Path("//10.128.169.11/DoeLab65TB") / Path(*image_path.parts[3:])
        assert image_path.exists()
    if display_num is None:
        display_num = scene_num
    viewer = Viewer(title=f"napari scene {display_num}")
    if image_path.suffix == ".czi":
        for data, metadata, _ in reader_function_with_args(
            image_path, scene_index=scene_num, next_scene_inds=[]
        ):
            add_image_out = viewer.add_image(data=data, **metadata)
            if isinstance(add_image_out, list):
                layers = add_image_out
            else:
                layers = [add_image_out]
        return viewer
    if image_path.suffix in (".tiff", ".tif"):
        with TiffFile(image_path) as tif:
            series = tif.series[scene_num]
            axes = series.get_axes()
            assert axes == "ZCYX"
            if tif.imagej_metadata is not None:
                zdim = tif.imagej_metadata.get("spacing", 1.)
            else:
                zdim = 1.
            first_page = series.pages[0]
            if first_page is None:
                raise ValueError("Could not read resolution")
            if isinstance(first_page, TiffFrame):
                raise ValueError("Could not read resolution")
            ydim = _xy_voxel_size(first_page.tags, "YResolution")
            xdim = _xy_voxel_size(first_page.tags, "XResolution")
            data = series.asarray()
        names = [f"raw-{f}-channel" for f in range(data.shape[1])]
        viewer.add_image(data, channel_axis=1, scale=(zdim, ydim, xdim), name=names)
        return viewer
    raise ValueError(f"{image_path.suffix} not known")


def bind_key(viewer: Viewer, views: Sequence[str]):
    """
    Binds the d key so that it toggles between the
    views
    """
    current_ind = 0

    def toggle_channel(*args):
        nonlocal current_ind
        _ = args
        set_view(viewer, views[current_ind])
        current_ind = (current_ind + 1) % len(views)

    viewer.bind_key(key="d", func=toggle_channel, overwrite=True)
    toggle_channel()


def set_view(viewer: Viewer, view_str: str):
    """
    sets the veiw by decoding the view string
    wich is color (gmk) or gone (_)
    asserts string length is the number of channels
    """
    for i, char in enumerate(view_str):
        layer = img_layer(viewer, i)
        if char == "_":
            layer.visible = False
        else:
            layer.colormap = COLOR_MAP[char]
            layer.visible = True


def add_fluor(fluor: str, known_fluors: list[str]) -> list[str]:
    """
    use user input to add a fluor to a flo
    """
    n_known_fluors = len(known_fluors)
    for i, this_fluor in enumerate(known_fluors):
        print(f"[{i}] {this_fluor}", end=" ", flush=False)
    print(f"[{n_known_fluors}]")
    position = int(input(f"Where do you want to insert {fluor}?\t"))
    if position == n_known_fluors:
        known_fluors.append(fluor)
    else:
        known_fluors.insert(position, fluor)
    set_fluorophoe_list(known_fluors)
    return known_fluors


def _get_fluorophers_image_series(viewer: Viewer) -> pd.Series:
    """
    returns a series with fluorophore name as key and a column with the viewer
    """
    img_layers = [layer for layer in viewer.layers if isinstance(layer, Image)]
    fluorophores_dict: dict[str, Image] = {}
    known_fluors = get_fluorophore_list()
    # add integers in case flour is not known
    int_padded_known_fluors = [str(i) for i in range(50)] + known_fluors
    for layer in img_layers:
        regex_match = re.match(f"^raw-(.+)-channel$", layer.name)
        if regex_match is None:
            continue
        fluor = regex_match.group(1)
        try:
            int(fluor)
        except:
            pass
        fluorophores_dict[fluor] = layer
        if fluor not in int_padded_known_fluors:
            add_fluor(fluor, known_fluors)
    if len(fluorophores_dict) == 0:
        raise ValueError("No czi channels found")
    # add new fluor to list
    out_series = pd.Series(fluorophores_dict, index=sorted(fluorophores_dict.keys(), key=int_padded_known_fluors.index))
    return out_series


def img_layer(viewer: Viewer, index: int) -> Image:
    """
    gets the index'th most red image layer
    """
    fluorophores = _get_fluorophers_image_series(viewer)
    return fluorophores.iloc[index]


def save_mip(viewer: Viewer, out_path: Path, zrange: tuple[int| None, int | None] = (None, None), view_str: str | None = None) -> np.array:
    """
    Saves a max intensity projection over z values specified in zrange to out_path
    """
    # clear the scene before adding new layers
    for layer in viewer.layers:
        layer.visible = False
    fluorophores = _get_fluorophers_image_series(viewer)
    if view_str is None:
        mip_layers: list[Image | None] = fluorophores.to_list()
    else:
        mip_layers = []
        for i, char in enumerate(view_str):
            if char == "_":
                mip_layers.append(None)
            else:
                mip_layers.append(img_layer(viewer, i))
    for i, layer in enumerate(mip_layers):
        if layer is None:
            continue
        assert layer.data.shape[0] == 1
        mip = layer.data[0, slice(*zrange), :, :].max(axis=0)
        if view_str is None:
            colormap = layer.colormap
        else:
            colormap = COLOR_MAP[view_str[i]]
        viewer.add_image(mip[np.newaxis, np.newaxis, :, :], scale=layer.scale, translate=layer.translate, colormap=colormap, blending="additive", name=f"mip_{layer.name}")
    # take image
    dims_step = list(viewer.dims.current_step)
    dims_step[1] = 0 
    viewer.dims.current_step = dims_step
    return viewer.screenshot(flash=False, path=out_path)


def burn_in_contrast(
    image: np.ndarray,
    contrast_min: float,
    contrast_max: float,
    knee: float | None = None,
) -> np.ndarray:
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
    image = image.astype(float_type)
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
    return out_array.astype(np.uint8)


class AnalysisStep(Protocol):
    """
    A step in analysis which returns the image data and adds it to the
    """

    def __call__(self, viewer: Viewer, layer: int | napari.layers.Layer, 
                 name: str | None, *args, **kwargs) -> ImageData:
        ...


def _make_analysis_step(
    function: Callable[..., ImageData],
    doc: str,
    out_type: Literal["Image", "Labels"] = "Image",
    additive=False,
) -> AnalysisStep:
    """
    takes a function that takes in image data and returns an AnalysisStep
    """

    def out_function(
        viewer: Viewer, layer: int | napari.layers.Layer, *args, name: str | None = None,  **kwargs, 
    ) -> LabelsData | ImageData:
        if name is None:
            name = function.__name__
        if isinstance(layer, napari.layers.Layer):
            pass
        elif layer >= 0:
            layer = img_layer(viewer, layer)
        else:
            layer = viewer.layers[layer]
        out_data = function(layer.data, *args, **kwargs)
        if out_type == "Image":
            new_layer = viewer.add_image(out_data, name=name)
            if additive:
                new_layer.additive = True
        elif out_type == "Labels":
            new_layer = viewer.add_labels(out_data, name=name)
            new_layer.contour = 1
        new_layer.translate = layer.translate
        new_layer.scale = layer.scale
        return out_data

    out_function.__doc__ = doc
    return out_function


def print_contrast_limits(viewer: Viewer):
    """
    prints out the contrast limits of the last image in a nice way
    """
    contrast_min, contrast_max = viewer.layers[-1].contrast_limits
    print(f"{(contrast_min, contrast_max)=}")


blur = _make_analysis_step(
    nsbatwm.gaussian_blur, "does a gaussian_blur with sigma as a kwarg"
)
contrast = _make_analysis_step(
    burn_in_contrast,
    "Makes a new layer with enhanced contrast kwargs: contrast_min: float, contrast_max: float, knee: float | None=None",
)
label = _make_analysis_step(
    nsbatwm.voronoi_otsu_labeling,
    "spot_sigma: float=2, outline_sigma: float=2",
    out_type="Labels",
)

threshold = _make_analysis_step(
    nsbatwm.threshold_otsu,
    "spot_sigma: float=2, outline_sigma: float=2",
    out_type="Labels",
)
tophat = _make_analysis_step(
    nsbatwm.white_tophat,
    "Does a white tophat to make more clear puncta: radius: float=2",
    out_type="Image",
    additive=True
)
