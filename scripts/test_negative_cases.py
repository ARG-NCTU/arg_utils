#! /usr/bin/env python3
import os
import pytest
import numpy as np
import add_path
import yaml
from numpy.linalg import inv
from arg_utils.camera_projection import camera_projection as cam_proj
from scipy.spatial.transform import Rotation as R

def test_invalid_image_path():
    test_cam_proj = cam_proj()
    with pytest.raises(Exception):  # Replace Exception with specific error if known
        test_cam_proj.read_images(300, 'invalid_path/rgb/', 'invalid_path/depth/')


def test_apriltag_detection_no_tags():
    test_cam_proj = cam_proj()
    test_cam_proj.read_images(301, 'datas/ViperX_apriltags/rgb/', 'datas/ViperX_apriltags/depth/')  # Assuming image 301 has no tags
    test_cam_proj.apriltag_detection()
    assert len(test_cam_proj.detection_results) == 0


def test_solvePnP_no_tags_detected():
    test_cam_proj = cam_proj()
    test_cam_proj.read_camera_info('datas/ViperX_apriltags/camera_info.yaml')
    test_cam_proj.read_images(301, 'datas/ViperX_apriltags/rgb/', 'datas/ViperX_apriltags/depth/')  # Assuming image 301 has no tags
    test_cam_proj.apriltag_detection()
    with pytest.raises(Exception):  # Replace Exception with specific error if known
        test_cam_proj.solvePnP(0.0415)


if __name__ == '__main__':
    pytest.main(['-s', '-v', __file__])
