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

import numpy as np
from napari_czifile2 import reader_function_with_args
from napari_czifile2.io import CZISceneFile
from napari.layers import Image
from napari.viewer import Viewer
from napari.types import ImageData, LabelsData, LayerData
import napari_segment_blobs_and_things_with_membranes as nsbatwm

FLUOROPHORE_LIST = ["AF405", "AF488", "AF546", "AF555", "AF647"]

COLOR_MAP = {"g": "green", "m": "magenta", "k": "gray", "r": "red", "p": "PiYG"}


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


def generate_random_key(key_path: Path, image_paths: list[Path]):
    """
    creates a list of tuples of
    (file_path (str), scene_num (int))
    and saves it to key_path for use in
    get_random_viewer
    """
    path_scenes: list[PathScene] = []
    for path in image_paths:
        n_scenes = CZISceneFile.get_num_scenes(path)
        path_scenes.extend(PathScene(path, i) for i in range(n_scenes))
    random.shuffle(path_scenes)
    key_path.write_text(json.dumps([ps.to_tuple() for ps in path_scenes], indent=4))


def get_random_viewer(key_path: Path, img_num: int) -> Viewer:
    """
    gets the viewer at img_num from key_path
    """
    this_path_scene = PathScene.from_tuple(
        json.loads(key_path.read_text("utf-8"))[img_num]
    )
    return get_viewer_at_czi_scene(this_path_scene.path, this_path_scene.scene)


def get_viewer_at_czi_scene(czi_file_path: Path, scene_num: int) -> Viewer:
    """
    gets a viewer with the czi file opened at scene
    """
    viewer = Viewer()
    for data, metadata, _ in reader_function_with_args(
        czi_file_path, scene_index=scene_num, next_scene_inds=[]
    ):
        viewer.add_image(data=data, **metadata)
    return viewer


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

    viewer.bind_key(key="d", func=toggle_channel)
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


def img_layer(viewer: Viewer, index: int):
    """
    gets the index'th most red image layer
    """
    img_layers = [layer for layer in viewer.layers if isinstance(layer, Image)]
    pattern = re.compile(r"S\d+\s(.*)-T\d+")
    fluorophores: dict[str, Image] = {}
    for layer in img_layers:
        match = pattern.match(layer.name)
        if match is None:
            # probaby not a czi file layer
            continue
        fluorophores[match.group(1)] = layer
    this_flurophore = sorted(fluorophores.keys(), key=FLUOROPHORE_LIST.index)
    return fluorophores[this_flurophore[index]]


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
    begining_slope = 255 * knee / contrast_min
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

    def __call__(self, viewer: Viewer, scene_index: int, *args, **kwargs) -> ImageData:
        ...


def _make_analysis_step(
    function: Callable[..., ImageData],
    doc: str,
    out_type: Literal["Image", "Labels"] = "Image",
) -> AnalysisStep:
    """
    takes a function that takes in image data and returns an AnalysisStep
    """

    def out_function(
        viewer: Viewer, layer_index: int, *args, **kwargs
    ) -> LabelsData | ImageData:
        if layer_index > 0:
            layer = img_layer(viewer, layer_index)
        else:
            layer = viewer.layers[layer_index]
        out_data = function(layer.data, *args, **kwargs)
        if out_type == "Image":
            new_layer = viewer.add_image(out_data, name=function.__name__)
        elif out_type == "Labels":
            new_layer = viewer.add_labels(out_data, name=function.__name__)
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
