"""
Some code to rotate the embryo so that the VNC is in the same position
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt


def rotate_using_coords(image: np.ndarray, point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """
    roates the image such that point1 is directly negative x from point2
    returns the rotated image
    """
    unorm_vector = point2 - point1
    y_axis = np.array([0, 1, 0])
    vector =  unorm_vector / np.linalg.norm(unorm_vector)
    cross_product = np.cross(vector, y_axis)
    dot_product = np.dot(vector, y_axis)
    norm_cross = np.linalg.norm(cross_product)
    axis = cross_product / norm_cross
    angle = np.arccos(dot_product)
    rotation = R.from_rotvec(axis * angle)
    rotation_matrix = rotation.as_matrix()
    rotated_image = affine_transform(image, rotation_matrix, output_shape=(20, 20, 20), order=1, offset=[-5, -5, -5])
    return rotated_image

def test_rotation():
    image = np.zeros((10, 10, 10))
    coords = np.array([1, 2, 5]), np.array([4, 3, 0])
    for coord in coords:
        image[coord[0], coord[1], coord[2]] = 1
    out_img = rotate_using_coords(image, *coords)
    # viewer = napari.Viewer()
    viewer.add_image(image)
    viewer.add_image(out_img)

