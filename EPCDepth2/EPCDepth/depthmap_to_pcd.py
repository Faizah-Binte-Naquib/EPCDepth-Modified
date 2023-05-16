import imageio.v3 as iio
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
import torch
from chamferdist import ChamferDistance

def compute_point_cloud(disp_path, matrix_path):
    images  = (glob.glob(disp_path))
    matrices = (glob.glob(matrix_path))
    cloud_points = []
    print("Here!")
    for image, matrix in tqdm(zip(images[:50],matrices[:50])):
        with open(matrix) as f:
            lines = f.readlines()
        
        if(len(lines[0].split(', '))>1):
            camera_params = lines[0].split(', ')
            # Camera parameters:
            FX_DEPTH = float(camera_params[0][1:])
            FY_DEPTH = float(camera_params[2])
            CX_DEPTH = float(camera_params[4])
            CY_DEPTH = float(camera_params[5])
        else:
            camera_params = lines[0].split(' ')
            # Camera parameters:
            FX_DEPTH = float(camera_params[0])
            FY_DEPTH = float(camera_params[2])
            CX_DEPTH = float(camera_params[4])
            CY_DEPTH = float(camera_params[5])

        # Read depth image:
        depth_image = iio.imread(image)

        # Compute depth grayscale:
        depth_grayscale = np.array(256 * depth_image / 0x0fff, dtype=np.uint8)
        iio.imwrite('depth_grayscale.png', depth_grayscale)


        # get depth image resolution:
        height = depth_image.shape[0]
        width = depth_image.shape[1]
        # compute indices:
        jj = np.tile(range(width), height)
        ii = np.repeat(range(height), width)
        # Compute constants:
        xx = (jj - CX_DEPTH) / FX_DEPTH
        yy = (ii - CY_DEPTH) / FY_DEPTH
        # transform depth image to vector of z:
        length = height * width
        z = depth_image[:,:,0].reshape(length)
        # compute point cloud
        pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
        cloud_points.append(pcd)

        # Convert to Open3D.PointCLoud:
        #pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
        #pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
        
        # Visualize:
        #o3d.visualization.draw_geometries([pcd_o3d])
    return cloud_points


if __name__ == '__main__':


    source_disp_path = '/mnt/c/Users/faiz2/OneDrive/Desktop/Classes/Vision Sensor/final project/EPCDepth2/EPCDepth/disparity/disps_vis2/*'
    source_matrix_path = '/mnt/c/Users/faiz2/OneDrive/Desktop/Classes/Vision Sensor/final project/EPCDepth2/EPCDepth/disparity/intrinsic/*'

    target_disp_path = '/mnt/c/Users/faiz2/OneDrive/Desktop/Classes/Vision Sensor/final project/EPCDepth2/EPCDepth/depth_selection/test_depth_completion_anonymous/image/*'
    target_matrix_path = '/mnt/c/Users/faiz2/OneDrive/Desktop/Classes/Vision Sensor/final project/EPCDepth2/EPCDepth/depth_selection/test_depth_completion_anonymous/intrinsics/*'

    source = compute_point_cloud(source_disp_path, source_matrix_path)
    target = compute_point_cloud(target_disp_path, target_matrix_path)

    #EVALUATE WITH THE GROUND TRUTH
    chamferDist = ChamferDistance()
    index=0
    for s,t in tqdm(zip(source,target)):
        s= torch.unsqueeze(torch.tensor(s), dim=0).cuda()
        t= torch.unsqueeze(torch.tensor(t), dim=0).cuda()
        dist_forward = chamferDist(s, t)
        index+=1
        print("For image:", index)
        print("Chamder Distance:",dist_forward.detach().cpu().item())