"""
Count the 25c Flies
"""

from pathlib import Path
from napari_3d_counter import Count3D, CellTypeConfig
import napari_scripts as ns

path = Path("/Users/peternewstein/Downloads/Data-backup/488cmyc555pmad647eve_25c.czi")


# manual for loop
i = 4

viewer = ns.get_viewer_at_czi_scene(path, i)
ns.bind_key(viewer, ["gm_", "__p"])

count_3d = Count3D(
    viewer,
    [
        CellTypeConfig("cMyc+eve+", out_of_slice_point_size=1, outline_size=3),
        CellTypeConfig("cMyc+eve-", out_of_slice_point_size=1, outline_size=3),
        CellTypeConfig("earlyUMN", out_of_slice_point_size=1, outline_size=3),
    ],
)
viewer.window.add_dock_widget(count_3d)

# Save the file when done
input("press enter to save")

count_3d.save_points_to_df().to_csv(path.parent / f"{i}.csv")
viewer.close()
