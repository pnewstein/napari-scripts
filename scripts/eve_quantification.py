"""
Quantify the eve expression in anonomized data
"""
from pathlib import Path

from napari_3d_counter import Count3D, CellTypeConfig
import napari_scripts as ns

quant_dir = Path().home() / "eve_quant"
rand_key_path = quant_dir / "key.json"

def anonomize_data():
    ns.generate_random_key(
        rand_key_path,
        [
            Path("/Users/peternewstein/Downloads/Data-backup/488cmyc555pmad647eve_25c.czi"),
            Path("/Users/peternewstein/Downloads/Data-backup/488cmyc555pmad647eve_31c.czi"),
        ],
    )


# manual for loop
i = 0
viewer = ns.get_random_viewer(rand_key_path, i)

count_3d = Count3D(
    viewer,
    [
        CellTypeConfig("umn", outline_size=20, out_of_slice_point_size=5),
        CellTypeConfig("accrp2", outline_size=20, out_of_slice_point_size=5),
        CellTypeConfig("el", outline_size=20, out_of_slice_point_size=5),
    ],
)
viewer.window.add_dock_widget(count_3d)

# Save the file when done
input("press enter to save")

count_3d.save_points_to_df().to_csv(f"{rand_key_path}_{i}.csv")
