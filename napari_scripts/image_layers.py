"""
code to retreve image layers
"""

import json
from pathlib import Path
import re

import pandas as pd
from napari.layers import Image
from napari.viewer import Viewer


FLUOROPHORE_LIST_PATH = Path(__file__).parent / "fluorophore_list.json"


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
        regex_match = re.match(r"^raw-(.+)-channel$", layer.name)
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
    out_series = pd.Series(
        fluorophores_dict,
        index=sorted(fluorophores_dict.keys(), key=int_padded_known_fluors.index),
    )
    return out_series


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


def img_layer(viewer: Viewer, index: int) -> Image:
    """
    gets the index'th most red image layer
    """
    fluorophores = _get_fluorophers_image_series(viewer)
    return fluorophores.iloc[index]
