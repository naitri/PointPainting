from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import struct
from mmseg.core.evaluation import get_palette,get_classes
import cv2
import matplotlib.pyplot as plt
import os




def label_to_color(semantic_map,palette):
    """
        @brief      Converting semantic label map into rgb cityscapes color palatte with respect to label ids
        @param      semantic_map (n*m)
        @param      palette By default it is cityscapes palette
        @return     colored semantic_map (n*m*3)
    """

    color_seg = np.zeros((semantic_map.shape[0], semantic_map.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[semantic_map  == label] = color
    
        
    #Converting BGR to RGB
    color_seg = color_seg[..., ::-1]
    color_seg = color_seg.astype(np.uint8)
    return color_seg




def semantics_to_colors(semantics:np.array,palette:np.array) -> np.array:
        """
        @brief      Converting semantic map into rgb cityscapes color palatte with respect to label names
        @param      semantics [npoints, 1]
        @return     Colors [npoints, 3]
        """
        
        
        colors = np.ones((semantics.shape[0], 3))
       
        for label,color in enumerate(palette):
                   
            
            colors[semantics == label] = (color[0]/255,color[1]/255,color[2]/255)
     
            
        return colors



def visuallize_pointcloud(pointcloud: np.array,path:str,filename:str,palette:np.array) -> None:
        """
        @brief      Visualizing colored point cloud
        @param      pointcloud  in lidar coordinate [npoints, 4] in format of [X Y Z label_ids]
        @return     None
        """
        
        #Get RGB values from pointcloud
        semantics  = pointcloud[:, 3]
        #Get xyz values from pointcloud
        xyz = pointcloud[:, 0:3]


        #Initialize Open3D visualizer
        visualizer = o3d.visualization.Visualizer()
        pcd = o3d.geometry.PointCloud()
        visualizer.add_geometry(pcd)


        # Get colors of each point according to cityscapes labels
        colors = semantics_to_colors(semantics,palette)
    
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join(path,"results","painted_cloud",filename+".pcd"),pcd)
        


def transform_velo_to_cam(R0, Tr_cam_to_lidar):
    """
        @brief      Implementation for getting Trasformation matrix from lidar to camera
        @param      R0: rectification rotation matrix
        @param      Tr_cam_to_lidar: Transformation matrix from camera to lidar [3,4]
        @return     Trasformation matrix from lidar to camera[3,4]
        """
    R_ref2rect = np.eye(4)
    R0_rect = R0.reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    R_ref2rect_inv = np.linalg.inv(R_ref2rect)  # rect2ref_cam

    # inverse rigid transformation
    cam2velo_ref = np.vstack((Tr_cam_to_lidar.reshape(3, 4), np.array([0., 0., 0., 1.])))  
    P_cam2velo_ref = np.linalg.inv(cam2velo_ref)

    proj_mat = P_cam2velo_ref @ R_ref2rect_inv
    return proj_mat

def projection_velo_to_cam(R0, Tr_lidar_to_cam,P):
    """
        @brief      Projection matrix for projection of lidar to camera
        @param      R0: Rectified Rotation Matrix
        @param      Tr_lidar_to_cam: Transformation matrix for lidar to camera
        @param      P: Perspective Intrinsics [3,4]
        @return     Projection matrix[3,4]
    """

    R_rect = np.eye(4)
    R0 = R0.reshape(3, 3)
    R_rect[:3, :3] = R0
    P_ = P.reshape((3, 4))
    proj_mat = P_ @ R_rect @ Tr_lidar_to_cam
    return proj_mat

def convert_bin_to_pcd(binary_file, pcd_filepath):
    """
        @brief      Coversion of bin file to pcd
        @param      binary_file
        @param      pcd_filepath: path to store pcd file
        @return     None
    """

    list_pcd = []
    size_float = 4
    with open(binary_file, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)

    o3d.io.write_point_cloud(pcd_filepath, pcd)

def quaternion_to_rotation(quat):
    """
        @brief      Coversion of quaternion to Rotation matrix
        @param      Quaternions [1,4]
        @return     Rotation matrix [3,3]
    """
    rot_matrix = Rotation.from_quat(quat)
    rot_matrix = rot_matrix.as_matrix()

    return rot_matrix



def convert_3D_to_2D(P,lidar_pts):
    """
        @brief      Projecting 3D points on the image
        @param      P lidar to camera projection matrix[3,4]
        @param      lidar_pts [npoints,3]
        @return     points on image(2D points)[npoints,2] and projected depth [npoints,1]
    """
    
    

    pts_3d = convert_3d_to_hom(lidar_pts)
    pts_2d= np.dot(pts_3d,P.T)
    
    depth = pts_2d[:, 2]
    depth[depth==0] = -1e-6

    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    pts_2d = pts_2d[:, :2]

    return pts_2d,depth


def remove_lidar_points_beyond_img(P,lidar_pts, xmin, ymin, xmax, ymax):
    """
        @brief      Filter lidar points, keep only those which lie inside image
        @param      P lidar to camera projection matrix[3,4]
        @param      lidar_pts [npoints,3]
        @param      xmin minimum image size width
        @param      ymin minimum image size height
        @param      xmax maximum image size width
        @param      ymax maximum image size height
        @return     points on image(2D points)[npoints,2], list of indices, projected depth [npoints,1]
    """
    pts_2d,depth = convert_3D_to_2D(P,lidar_pts)
  
    inside_pts_indices = ((pts_2d[:, 0] >= xmin) & (pts_2d[:, 0] < xmax) & (pts_2d[:, 1] >= ymin) & (pts_2d[:, 1] < ymax))

    
   
    return  pts_2d, inside_pts_indices,depth


def project_lidar_on_image(P, lidar_pts, size):
    """
        @brief      Projecting 3D lidar points on the image
        @param      P lidar to camera projection matrix[3,4]
        @param      lidar_pts [npoints,3]
        @param      size: image size
        @return     filtered points on image(2D points)[npoints,2] and  projected depth [npoints,1]
    """
    all_pts_2d, fov_inds, depth = remove_lidar_points_beyond_img(P,lidar_pts, 0, 0,size[0], size[1])

    return all_pts_2d[fov_inds],depth[fov_inds], lidar_pts[fov_inds]

def convert_3d_to_hom(pts_3d):
    """
        @brief      Converting lidar points into homogenous coordinate
        @param      pts_3d [npoints,3]
        @return     pts_3d into homogenous coordinate [npoints,4]
    """
   
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom