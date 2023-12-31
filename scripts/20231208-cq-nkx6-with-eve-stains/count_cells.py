from pathlib import Path
import pickle


import napari
from napari_3d_counter import Count3D, CellTypeConfig
from numpy.typing import NDArray
import numpy as np
from tifffile import imread
import pyclesperanto_prototype as cle  # version 0.24.1
import napari_segment_blobs_and_things_with_membranes as nsbatwm

from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
from collections.abc import Mapping
import pickle

from numpy.typing import NDArray
import tifffile
import napari
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import numpy as np
import pyclesperanto_prototype as cle  # version 0.24.1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

files = sorted(Path("/Users/petern/Documents/tmp/cq-eve/").glob("*.tif"))


def load_file(name: Path, viewer: napari.viewer.Viewer) -> napari.viewer.Viewer:
    """
    loads the 4 channel hdtf file
    """
    img = tifffile.imread(name)
    viewer.add_image(img, channel_axis=1, name=["pmad", "cmyc", "eve"])
    viewer.layers["pmad"].colormap = "magenta"
    viewer.layers["cmyc"].colormap = "green"
    viewer.layers["eve"].colormap = "gray"
    viewer.layers["eve"].visible = False
    return viewer


def bind_key(viewer: napari.viewer.Viewer):
    def toggle_all_chans(*args):
        _ = args
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.visible = not layer.visible

    viewer.bind_key(key="d", func=toggle_all_chans)


def main(file_num: int):
    file_name = files[file_num]
    viewer = napari.viewer.Viewer()
    load_file(file_name, viewer)
    count_3d = Count3D(
        viewer,
        [
            CellTypeConfig("cMyc+eve+"),
            CellTypeConfig("cMyc+eve-"),
            CellTypeConfig("earlyUMN"),
        ],
    )
    bind_key(viewer=viewer)
    viewer.window.add_dock_widget(count_3d)
    print(file_name)
    input("press enter to save")

    count_3d.save_points_to_df().to_csv(file_name.with_suffix(".csv"))


if __name__ == "__main__":
    # manual for loop
    main(2)
