

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette,get_classes
import numpy as np
import open3d as o3d
import os
from datetime import date
import calibration as ca
import utils as ut
import matplotlib.pyplot as plt
import natsort
from argparse import ArgumentParser
import cv2
import matplotlib.pyplot as plt
import sys
from scipy.spatial import KDTree
from distutils.util import strtobool

def main():

    """
    @brief      Generate semantically segmented images, point clouds, and inpainted images and point cloud.
    @details    It uses SegFormer network for segmentation and hence weight file and dependent files of SegFormer needs to be arranged as per requirements. Also the calibration file, calibration file parser
                should be in the same path folder

                Example usage:
                $ python point_paint.py /home/nrajyagu/Documents/SegFormer local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py 
                /home/nrajyagu/Documents/SegFormer/segformer.b5.1024x1024.city.160k.pth --device cuda:0 --palette cityscapes

    """
    parser = ArgumentParser()
    parser.add_argument('--path', help='path file for data')
    parser.add_argument('--config', help='Config file of SegFormer')
    parser.add_argument('--checkpoint', help='Checkpoint file of SegFormer')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')

    args = parser.parse_args()

    #Get path for data folder containing rgb and pcd files
    data_path = os.path.join(args.path, "data")
    rgb_img_dir = os.path.join(data_path, "rgb")
    lidar_dir = os.path.join(data_path, "fused_pcd")
    

    #Get all calibration parameters
    calib = ca.CalibrationData(os.path.join(args.path, "calib.txt"))
    R = calib.R0
    P = calib.P
    K = calib.K
    D = calib.D
    Tr_cam_to_lidar = calib.Tr_cam_to_lidar
    Tr_lidar_to_cam = ut.transform_velo_to_cam(R, Tr_cam_to_lidar)
    P_lidar_to_cam = ut.projection_velo_to_cam(R, Tr_lidar_to_cam,P)



    #initialize SegFormer model and classes
    print("Initializing the pipeline")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    palette = get_palette(args.palette)
    classes = get_classes(args.palette)
    for files in os.walk(rgb_img_dir):
        files =files[2]
        for filename in natsort.natsorted(files,key=str):
               
               

                '''
                #############################################Segmented image generation#######################################
                '''
                rgb_img = cv2.imread(os.path.join(rgb_img_dir, filename))
                fused_img = rgb_img.copy()
                result = inference_segmentor(model, rgb_img)
                seg_img = result[0]
                segmented_img = ut.label_to_color(seg_img,palette)

                cv2.imwrite(os.path.join(args.path,"results", "segmentation",filename),segmented_img)

                

                '''
                #############################################Point cloud generation#######################################
                '''
                file = os.path.join(lidar_dir, filename[:-4] + ".bin")

                pcd_file = ut.convert_bin_to_pcd(file,os.path.join(lidar_dir, filename[:-4] + ".pcd"))
                point_cloud = np.asarray(o3d.io.read_point_cloud(os.path.join(lidar_dir, filename[:-4] + ".pcd")).points)
                
                #3d points infront of camera will only be projected
                idx = point_cloud[:,0] >= 0
                point_cloud = point_cloud[idx]

                pts_2D,depth, pts_3D_img = ut.project_lidar_on_image(P_lidar_to_cam, point_cloud, (rgb_img.shape[1], rgb_img.shape[0]))

                #Number of lidar points projected on image
                N = pts_3D_img.shape[0]

                #Creating semantic channel for point cloud
                semantic = np.zeros((N,1), dtype=np.float32)
                
                for i in range(pts_2D.shape[0]):
                    if i >= 0:

                        x = np.int32(pts_2D[i, 0])
                        y = np.int32(pts_2D[i, 1])

                        classID = np.float64(segmented_img[y, x]) 
                       
                        pt = (x,y)
                        cv2.circle(fused_img, pt, 2, color=tuple(classID), thickness=1)

                        semantic[i] = seg_img[y,x]
                    
                stacked_img = np.vstack((rgb_img, segmented_img,fused_img))
                cv2.imwrite(os.path.join(args.path, "results","projected",filename),stacked_img)


                rgb_pointcloud = np.hstack((pts_3D_img[:,:3], semantic))


                
                
                ut.visuallize_pointcloud(rgb_pointcloud,args.path,filename[:-4],palette)
                
                
if __name__ == '__main__':
    main()
