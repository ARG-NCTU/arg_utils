# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_utils.ipynb.

# %% auto 0
__all__ = ['gdown_unzip', 'gdown_download', 'pose_dis', 'waypoint']

# %% ../00_utils.ipynb 4
import os
import sys
import gdown
import copy
import math
from zipfile import ZipFile

def gdown_unzip(id, filename):
    """download a zipfile and unzip it
    """
    dataset_url = 'https://drive.google.com/u/1/uc?id=' + id
    dataset_name = filename

    if not os.path.isdir(dataset_name):
        gdown.download(dataset_url, output = dataset_name + '.zip', quiet=False)
        zip_file = ZipFile( dataset_name + '.zip')
        #zip_file.extractall()
        zip_file.extractall() # depends on how to zip it
        zip_file.close()

def gdown_download(id, filename):
    """download a file
    """
    dataset_url = 'https://drive.google.com/u/1/uc?id=' + id
    dataset_name = filename

    if not os.path.isdir(dataset_name):
        gdown.download(dataset_url, output = dataset_name, quiet=False)
        
def pose_dis(pose_1, pose_2):
    """Compute distance between pose_1 and pose_2
    """
    x = pose_1[0] - pose_2[0]
    y = pose_1[1] - pose_2[1]
    z = pose_1[2] - pose_2[2]

    dis = math.sqrt(x**2+y**2+z**2)

    return dis

def waypoint(current_pose, Target_pose):
    """Generate a list of way points from current pose to target pose

    Input : current pose, target pose : list [x_pos, y_pos, z_pos, x_ori, y_ori, z_ori, w_ori]
    Return : a list of way points

    """
    waypoint_list = []
    factor = 0.5
    sub_pose = copy.deepcopy(current_pose)

    # threshold : distance between sub_pose and target_pose = 0.05 meter
    dis = pose_dis(sub_pose, Target_pose)
    while dis > 0.05:
        sub_pose[0] = (sub_pose[0] + Target_pose[0])*factor
        sub_pose[1] = (sub_pose[1] + Target_pose[1])*factor
        sub_pose[2] = (sub_pose[2] + Target_pose[2])*factor
        sub_pose[3] = Target_pose[3]
        sub_pose[4] = Target_pose[4]
        sub_pose[5] = Target_pose[5]
        sub_pose[6] = Target_pose[6]

        dis = pose_dis(sub_pose, Target_pose)

        waypoint_list.append(copy.deepcopy(sub_pose))

    waypoint_list.append(Target_pose)

    return waypoint_list
