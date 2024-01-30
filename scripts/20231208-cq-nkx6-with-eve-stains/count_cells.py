"""
make some screenshots for example images
"""
from pathlib import Path

from napari_3d_counter import Count3D, CellTypeConfig
import napari_scripts as ns

# Randomizes an order of reading scenes from multiple czi files
path = Path("/Volumes/DoeLab65TB/lab_member_data/Peter Newstein/data/20231208-check_cq_nkx6_with_eve_staining/488cmyc555pmad647eve_25c_high_res.czi")
# Reads the i'th random file from the rand_key_path
i = 0
viewer = ns.get_viewer_at_czi_scene(path, i)

