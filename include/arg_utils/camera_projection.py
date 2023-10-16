# AUTOGENERATED! DO NOT EDIT! File to edit: ../01_camera_projection.ipynb.

# %% auto 0
__all__ = ['camera_projection']

# %% ../01_camera_projection.ipynb 4
import numpy as np
import scipy as sp
import cv2
from cv2 import aruco
import apriltag
import time
import yaml

import pytransform3d.rotations as pr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import sys
import gdown
from zipfile import ZipFile

from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv

# %% ../01_camera_projection.ipynb 5
class camera_projection:
    def __init__(self, tag_s=0.0415):
        """init a camera projection object with default arguments
        """
        self.img_path = 'ViperX_apriltags/rgb/'
        self.depth_path = 'ViperX_apriltags/depth/'
        self.tag_size = tag_s
        self.s = 0.5 * self.tag_size

    def read_camera_info(self, cam_info_path = 'ViperX_apriltags/camera_info.yaml'):
        """read camera info from yaml file, path is given, camera info contains camera matrix and dist coefts
        """
        with open(cam_info_path, "r") as stream:
            try:
                camera_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.camera_matrix = np.array(camera_data['camera_matrix']['data'])
        self.camera_matrix = self.camera_matrix.reshape(3, 3)
        self.dist_coeffs = np.array(camera_data['distortion_coefficients']['data'])
        self.dist_coeffs = self.dist_coeffs.reshape(1, 5)
        self.cameraParams_Intrinsic = [self.camera_matrix[0,0], self.camera_matrix[1,1],
                                       self.camera_matrix[0,2], self.camera_matrix[1,2]]
 
    def read_images(self, idx=300, img_path = 'ViperX_apriltags/rgb/', depth_path = 'ViperX_apriltags/depth/'):
        """this function will load an image depend on the id number, often used in a for loop
        """
        img_path = img_path + str(idx) + '.png'
        depth_path = depth_path + str(idx) + '.png'
        self.img = cv2.imread(img_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.depth = cv2.imread(depth_path, -cv2.IMREAD_ANYDEPTH)
        self.img_dst = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def apriltag_detection(self, tag_type = 'tag36h11')):
        """detect if there is apriltag or not in self.gray, the self image
        """
        print("[INFO] detecting AprilTags...")
        options = apriltag.DetectorOptions(families= tag_type)
        detector = apriltag.Detector(options)
        #results = detector.detect(gray)
        self.detection_results, dimg = detector.detect(self.gray, return_image=True)
        print("[INFO] {} total AprilTags detected".format(len(self.detection_results)))

    def solvePnP(self):
        """this function will output the rotation matrix r_vec and translation matrix t_vex, this two matrixs is important for projection
        """
        img_pts = self.detection_results[0].corners.reshape(1,4,2)
        obj_pt1 = [-self.s, -self.s, 0.0]
        obj_pt2 = [ self.s, -self.s, 0.0]
        obj_pt3 = [ self.s,  self.s, 0.0]
        obj_pt4 = [-self.s,  self.s, 0.0]
        obj_pts = obj_pt1 + obj_pt2 + obj_pt3 + obj_pt4
        obj_pts = np.array(obj_pts).reshape(4,3)
        
        _, self.r_vec, self.t_vec = cv2.solvePnP(obj_pts, img_pts, self.camera_matrix,
                                       self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        R_mat, _ = cv2.Rodrigues(self.r_vec)
        T = np.hstack((R_mat, self.t_vec)).reshape(3,4)
        tag_pose = np.vstack((T, [0,0,0,1])).reshape(4,4)
        dist = np.linalg.norm(self.t_vec)
        
    def draw_point(self, tag_2_inv, base2joint):
        """for visualization, draw the project points on the image is important
        """
        # --------------- project a point ---------------
        tag2joint = np.matmul(tag_2_inv, base2joint)
        obj_pts = np.array([tag2joint[0,3], tag2joint[1,3], tag2joint[2,3]]).reshape(1,3)
        proj_img_pts, jac = cv2.projectPoints(obj_pts, self.r_vec, self.t_vec,
                                              self.camera_matrix, self.dist_coeffs)
        proj_img_pts = np.array(proj_img_pts).reshape(2,1)
        # --------------- draw a point ---------------
        draw_image = cv2.circle(self.img_dst, (int(proj_img_pts[0]), int(proj_img_pts[1])),
                                radius=5, color=(255, 0, 0), thickness=-1)
        return draw_image
