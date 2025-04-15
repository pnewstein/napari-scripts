"""
Some scripts for managing napari
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence
from dataclasses import dataclass
import random
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from napari_czifile2 import reader_function_with_args
from napari_czifile2.io import CZISceneFile
from napari.layers import Image, Labels
from napari.viewer import Viewer
import napari
from tifffile import TiffFile, TiffFrame
from skimage.segmentation import watershed
from magicgui.widgets import FunctionGui

from napari_scripts.image_layers import img_layer, _get_fluorophers_image_series

# export analysis_steps
from napari_scripts.analysis_steps import (
    blur,
    median,
    contrast,
    find_blobs,
    threshold,
    tophat,
    binary_dilation,
    binary_erosion,
    within_membranes,
    clear_edges,
    merge_dim_edged_labels,
    local_minima_watershed,
    remove_labels_enclaves,
    manual_thresh,
    local_maxima_to_points,
    get_npix_at_label,
    quantify_points_in_labels,
)


COLOR_MAP = {
    "g": "green",
    "m": "magenta",
    "k": "gray",
    "r": "red",
    "p": "PiYG",
    "c": "cyan",
    "y": "yellow",
}


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


def combine_masks(mask_layers: Iterable[Labels]) -> np.ndarray:
    """
    zip all masks together bitwise
    """
    mask_data = [l.data.squeeze().astype(bool) for l in mask_layers]
    shape = mask_data[0].shape
    assert all(d.shape == shape for d in mask_data)
    out = np.zeros(shape).astype(np.uint8)
    for i, data in enumerate(mask_data):
        out[data] |= 1 << i
    return out


def load_masks(condenced_masks: np.ndarray, names: list[str], **kwargs) -> list[Labels]:
    """
    kwargs are passed to labels
    """
    out: list[Labels] = []
    for i, name in enumerate(names):
        out.append(
            Labels(
                data=((condenced_masks & 1 << i) != 0).astype(np.uint8),
                name=name,
                **kwargs,
            )
        )
    return out


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
        path = catch_lab_server_paths(path)
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
    return get_viewer_from_file(
        this_path_scene.path, this_path_scene.scene, display_num=img_num
    )


def catch_lab_server_paths(path: Path) -> Path:
    """
    if a path is pointing to the unix lab server, converts to the lab server from windows
    else does nothing
    """
    if path.parts[:3] == ("\\", "Volumes", "DoeLab65TB") and os.name == "nt":
        image_path = Path("//10.128.169.11/DoeLab65TB") / Path(*path.parts[3:])
        assert image_path.exists()
        return image_path
    return path


def get_viewer_from_file(
    image_path: Path, scene_num: int, display_num=None, display=True
) -> Viewer:
    """
    gets a viewer with the file opened at scene. tries reading czis or tiffs
    """

    def _xy_voxel_size(tags, key):
        assert key in ["XResolution", "YResolution"]
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.0

    image_path = catch_lab_server_paths(image_path)
    if display_num is None:
        display_num = scene_num
    viewer = Viewer(title=f"napari scene {display_num}", show=display)
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
    fluor_names: list[str] | None = None
    if image_path.suffix in (".tiff", ".tif"):
        with TiffFile(image_path) as tif:
            series = tif.series[scene_num]
            axes = series.get_axes()
            if tif.imagej_metadata is not None:
                zdim = tif.imagej_metadata.get("spacing", 1.0)
            else:
                zdim = 1.0
            first_page = series.pages[0]
            if first_page is None:
                raise ValueError("Could not read resolution")
            if isinstance(first_page, TiffFrame):
                raise ValueError("Could not read resolution")
            ydim = _xy_voxel_size(first_page.tags, "YResolution")
            xdim = _xy_voxel_size(first_page.tags, "XResolution")
            data = series.asarray()
            # try getting channel names
            if tif.ome_metadata is not None:
                mdata = ET.fromstring(tif.ome_metadata)
                image = mdata[scene_num]
                namespace = {"ome": next(iter(mdata.attrib.values())).split()[0]}
                pixels = image.find("ome:Pixels", namespaces=namespace)
                assert pixels is not None
                channel_xml_mdatata = pixels.findall(
                    "ome:Channel", namespaces=namespace
                )
                fluor_names = [c.attrib["Name"] for c in channel_xml_mdatata]
        if fluor_names is None:
            names = [f"raw-{f}-channel" for f in range(data.shape[1])]
        else:
            names = [f"raw-{n}-channel" for n in fluor_names]

        if axes == "ZCYX":
            viewer.add_image(
                data,
                channel_axis=1,
                scale=(zdim, ydim, xdim),
                name=names,
                metadata={"scene_index": scene_num},
            )
        elif axes == "ZYX":
            viewer.add_image(
                data,
                scale=(zdim, ydim, xdim),
                name=names[0],
                metadata={"scene_index": scene_num},
            )
        elif axes == "CZYX":
            viewer.add_image(
                data,
                channel_axis=0,
                scale=(zdim, ydim, xdim),
                name=names,
                metadata={"scene_index": scene_num},
            )
        else:
            raise NotImplementedError(axes)
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

    viewer.bind_key(key_bind="d", func=toggle_channel, overwrite=True)
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


def save_mip(
    viewer: Viewer,
    out_path: Path,
    zrange: tuple[int | None, int | None] = (None, None),
    view_str: str | None = None,
) -> np.array:
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
        viewer.add_image(
            mip[np.newaxis, np.newaxis, :, :],
            scale=layer.scale,
            translate=layer.translate,
            colormap=colormap,
            blending="additive",
            name=f"mip_{layer.name}",
        )
    # take image
    dims_step = list(viewer.dims.current_step)
    dims_step[1] = 0
    viewer.dims.current_step = dims_step
    return viewer.screenshot(flash=False, path=str(out_path))


def watershed_merge_or_split_labels(
    labels_layer: napari.layers.Labels,
    points_layer: napari.layers.Points,
    image: napari.types.ImageData,
):
    """
    takes two poins and merge or split lables
    """
    points = points_layer.data
    labels = labels_layer.data
    if len(points) != 2:
        print("Wrong number of points")
        return
    points = points.astype(np.int64)
    lbl1, lbl2 = labels[points[:, 0], points[:, 1], points[:, 2]]
    lbl1_mask = labels == lbl1
    if lbl1 != lbl2:
        # merge them together
        labels[lbl1_mask] = lbl2
        labels_layer.data = labels
        points_layer.data = np.array([])
        return
    markers = np.zeros(labels.shape)
    for i, (px, py, pz) in enumerate(points, 1):
        markers[px, py, pz] = i
    new_neurons = watershed(image, markers=markers.astype(labels.dtype), mask=lbl1_mask)
    new_cell_mask = new_neurons == 1
    if new_cell_mask.sum() == 1 or (new_neurons == 2).sum() == 1:
        # failed try on binarized image
        new_neurons = watershed(
            lbl1_mask, markers=markers.astype(labels.dtype), mask=lbl1_mask
        )
        new_cell_mask = new_neurons == 1
    # get a new label
    currently_used_labels = np.unique(np.array(labels))
    possible_labels = np.arange(
        currently_used_labels[-1] + 2
    )  # get zero : 1+labels.max()
    next_label = np.where(~np.isin(possible_labels, currently_used_labels))[0][0]
    assert next_label not in currently_used_labels
    labels[new_cell_mask] = next_label
    labels_layer.data = labels
    points_layer.data = np.array([])


class WatershedMergeOrSplitLabels(FunctionGui):
    def __init__(self):
        super().__init__(watershed_merge_or_split_labels, call_button=True)
