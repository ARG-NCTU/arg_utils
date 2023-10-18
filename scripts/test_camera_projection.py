import os
import pytest
import numpy as np
import add_path
import matplotlib.pyplot as plt
import yaml
from numpy.linalg import inv
from arg_utils.camera_projection import camera_projection as cam_proj
from scipy.spatial.transform import Rotation as R
import wget
from zipfile import ZipFile

def wget_unzip(url, filename):
    if not os.path.isdir(filename):
        site_url = url
        file_name = wget.download(site_url)
        print(file_name)
        zip_file = ZipFile(file_name)
        zip_file.extractall(os.path.dirname(filename))
        zip_file.close()
        os.replace("./ViperX_apriltags.zip", "datas/ViperX_apriltags.zip")



def test_download():
    wget_unzip("ftp://140.113.148.83/arg-projectfile-download/arg_utils/ViperX_apriltags.zip","datas/ViperX_apriltags")
    
def test_read_camera_info():
    test_cam_proj = cam_proj()
    test_cam_proj.read_camera_info('datas/ViperX_apriltags/camera_info.yaml')
    assert np.array_equal(test_cam_proj.dist_coeffs, np.array([[0, 0, 0, 0, 0]]))
    assert test_cam_proj.cameraParams_Intrinsic == [615.7344970703125, 616.2518310546875, 317.8375854492188, 241.1808624267578]

def test_read_images():
    test_cam_proj = cam_proj()
    test_cam_proj.read_images(300, 'datas/ViperX_apriltags/rgb/', 'datas/ViperX_apriltags/depth/')
    assert test_cam_proj.img is not None
    assert test_cam_proj.gray is not None
    assert test_cam_proj.depth is not None
    assert test_cam_proj.img_dst is not None

def test_apriltag_detection():
    test_cam_proj = cam_proj()
    test_cam_proj.read_images(300, 'datas/ViperX_apriltags/rgb/', 'datas/ViperX_apriltags/depth/')
    expected_num_tags = 1
    test_cam_proj.apriltag_detection()
    assert len(test_cam_proj.detection_results) == expected_num_tags

def test_solvePnP():
    test_cam_proj = cam_proj()
    test_cam_proj.read_camera_info('datas/ViperX_apriltags/camera_info.yaml')
    test_cam_proj.read_images(idx=300, img_path='datas/ViperX_apriltags/rgb/', depth_path='datas/ViperX_apriltags/depth/')
    test_cam_proj.apriltag_detection('tag36h11')
    test_cam_proj.solvePnP(0.0415)
    test_r_vec = [round(i,2) for i in np.array(test_cam_proj.r_vec).ravel()]
    expect_r_vec = [ round(i,2) for i in [0.12234089, 0.100859, -0.24740067]]
    test_t_vec = [round(i,2) for i in np.array(test_cam_proj.t_vec).ravel()]
    expect_t_vec = [ round(i,2) for i in [-0.16983277, -0.03830798, 1.04091044]]
    assert test_r_vec == expect_r_vec
    assert test_t_vec == expect_t_vec

def test_draw_point():
    test_cam_proj = cam_proj()
    test_cam_proj.read_camera_info('datas/ViperX_apriltags/camera_info.yaml')
    test_cam_proj.read_images(idx=300, img_path='datas/ViperX_apriltags/rgb/', depth_path='datas/ViperX_apriltags/depth/')
    test_cam_proj.apriltag_detection('tag36h11')
    test_cam_proj.solvePnP(0.0415)

    r = R.from_euler('x', 180, degrees=True)
    r_tran = np.identity(4)
    r_tran[:3,:3] = r.as_matrix()
    with open('datas/ViperX_apriltags/pose/' + str(test_cam_proj.idx) + '.yaml', "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    tag_2 = np.array(data['Tag_2_pose'][0]['transformation_matrix'][0])
    print(tag_2)
    tag_2 = tag_2.reshape(4, 4)
    tag_2_inv = np.matmul(tag_2, r_tran)
    tag_2_inv = inv(tag_2_inv)

    draw_image = test_cam_proj.draw_point(tag_2_inv, np.identity(4))
    assert draw_image is not None
    # figsize = 15 # param larger is bigger, adjust as needed
    # plt.rcParams['figure.figsize'] = (figsize, figsize)
    # plt.imshow(draw_image, cmap = 'brg')
    # plt.show()
