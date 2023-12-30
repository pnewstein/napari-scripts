"""
Contains templates for use of the functions in napari-scripts
"""
from pathlib import Path

from napari_3d_counter import Count3D, CellTypeConfig
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import napari_scripts as ns

# Randomizes an order of reading scenes from multiple czi files
rand_key_path = Path("rand_key.json")
ns.generate_random_key(
    rand_key_path,
    [
        Path("/Users/peternewstein/Downloads/Data-backup/488cmyc555pmad647eve_25c.czi"),
        Path("/Users/peternewstein/Downloads/Data-backup/488cmyc555pmad647eve_31c.czi"),
    ],
)

# Reads the i'th random file from the rand_key_path
i = 0
viewer = ns.get_random_viewer(rand_key_path, i)

# set up the keybindings for the different views
ns.bind_key(viewer, ["gm_", "__k"])

# add a 3d counter
count_3d = Count3D(
    viewer,
    [
        CellTypeConfig("cq+eve+", outline_size=20, out_of_slice_point_size=5),
        CellTypeConfig("cq-eve+", outline_size=20, out_of_slice_point_size=5),
        CellTypeConfig("cq+eve-", outline_size=20, out_of_slice_point_size=5),
    ],
)
viewer.window.add_dock_widget(count_3d)

# Save the file when done
input("press enter to save")

count_3d.save_points_to_df().to_csv(rand_key_path.parent / f"{i}.csv")

## Workflow for automatic cell counting
blured = ns.blur(viewer, 1, sigma=1)

(contrast_max, contrast_min) = (0.3174772459464804, 0.07595189905254729)
contrasted = ns.contrast(
    viewer, -1, contrast_min=contrast_min, contrast_max=contrast_max
)
labels = ns.label(viewer, -1, spot_sigma=2.5, outline_sigma=2)
