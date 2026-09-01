"""
quantifies mouse fish
"""

from pathlib import Path

import napari
import napari_scripts as ns


def main():
    viewer = ns.get_viewer_from_file(
        Path(
            "/Volumes/DoeLab65TB/lab_member_data/Chundi Xu/peter/Well2_405WGA_488Vsx2_546Vsx1_647Lhx4.tif"
        ),
        0,
    )
    layer = ns.img_layer(viewer, 0)
    tophat = ns.tophat(viewer, layer, size=2)
    cell_bodies_and_edge = ns.within_membranes(
        viewer, tophat, spot_sigma=0.8, outline_sigma=0.2
    )
    cell_bodies = ns.clear_edges(viewer, cell_bodies_and_edge, top_bottom=False)
    real_cell_bodies = ns.merge_dim_edged_labels(
        viewer, cell_bodies, image=tophat.data, blur_sigma=0.8
    )
    eroded = ns.binary_erosion(viewer, real_cell_bodies, size=1.2)

    for layer in [ns.img_layer(viewer, i) for i in range(1, 4)]:
        tophat = ns.tophat(viewer, layer, size=1)
        psf.estimate_psf(tophat.data, 10, verbose=True)
