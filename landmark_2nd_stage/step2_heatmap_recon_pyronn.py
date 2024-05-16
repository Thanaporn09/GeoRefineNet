import argparse
import pyronn_torch
import numpy as np
import torch
from skimage.io import imsave
from filters import ram_lak_3D
from types import SimpleNamespace
import os
from tifffile import tifffile
import pandas as pd
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def main(config):
    if not os.path.exists(config.result_output_path):
        os.makedirs(config.result_output_path) 
    if config.condition == 'test':
        patient_test = ['female_3_head', 'female_4_head', 'female_6_head', 'female_7_head', 'female_10_head', 'male_3_head', 'male_4_head', 'male_7_head', 'male_12_head']
    elif config.condition == 'train':
        patient_test = ['female_1_head', 'female_5_head', 'female_8_head', 'female_9_head', 'female_11_head', 'female_12_head', 'female_13_head', 'female_14_head', 'female_15_head',
                       'male_1_head', 'male_2_head', 'male_5_head', 'male_6_head', 'male_8_head', 'male_9_head', 'male_10_head', 'male_11_head', 'male_13_head']

    ### load patient volume ###
    volume = pd.read_csv(config.img_info, header=None)     # (volume_size,header=None)
    for patient in patient_test:
        print("Calculating on {}...".format(patient))
        f = open(config.result_output_path + "{}.csv".format(patient), "w", newline="")
        writer = csv.writer(f)
        result = []
        volume_pat = volume[3][volume[0]==patient].values[0]
        x_spacing, y_spacing, z_spacing = 0.5, 0.5, 1
        x_ori, y_ori, z_ori = -199.75, -199.75, (((volume_pat*0.5)-0.5)*(-1))
        projection_matrices_recon = np.zeros((360, 3, 4))         # loading projection images and geometry information from file
        
        for i in range(360):
            proj_patient_path = config.projM_path + patient+'.txt'
            proj = np.loadtxt(proj_patient_path, delimiter=' ')
            p_i = proj[i]
            R = np.array([[p_i[0],p_i[1],p_i[2]], [p_i[4],p_i[5],p_i[6]], [p_i[8],p_i[9],p_i[10]]])
            T = np.array([[p_i[3]], [p_i[7]], [p_i[11]]])
            K = np.array([[1,0,0], [0,1,0], [0,0,1]])
            proj = K @ np.concatenate((R, T), axis=1)
            projection_matrices_recon[i,...] = proj

        # Reconstuction pipeline - ramlak filtering using PYRO-NN implementation of ramlak filter in Fourier domain
        geometry = {'detector_shape': (config.height_2d, config.width_2d),
                    'detector_spacing': (1.0, 1.0),
                    'number_of_projections': 360}
        geometry = SimpleNamespace(**geometry)
        ram_lak_filter = ram_lak_3D(geometry)
        projector_recon = pyronn_torch.ConeBeamProjector(
            (volume_pat, config.xy_voxel_3d, config.xy_voxel_3d),  # volume shape
            (x_spacing, y_spacing, z_spacing),  # volume spacing in mm
            (x_ori,y_ori,z_ori),  # volume origin in mm
            (360, config.height_2d, config.width_2d),  # projection_shape (number of projections, height, width)
            (1, 1),  # projection_spacing in mm
            (0, 0),  # projection origin in mm
            projection_matrices_recon
        )

        for i in range(11):
            point_wise_result = [] 
            # Load forward projection
            tif_file_path = os.path.join(config.tif_path, '{}/point_{}.tif'.format(patient.split('_head')[0], i))
            tif_image = tifffile.imread(tif_file_path)
            # Projection_images
            n = tif_image.shape[-1]
            ram_lak_filter = ram_lak_filter[:, :, :int(np.floor(n / 2) + 1)]
            projection_images = np.fft.rfft(tif_image)
            projection_images = np.multiply(projection_images, ram_lak_filter)
            projection_images = np.fft.irfft(projection_images, n=n)
            projection_images = torch.from_numpy(projection_images)
            recon = projector_recon.project_backward(projection_images, use_texture=True)
            # Find the coordinates of the maximum value in the image
            max_value_coordinates = np.unravel_index(np.argmax(recon.cpu().numpy()), recon.cpu().numpy().shape)
            point_wise_result.append(max_value_coordinates[2])
            point_wise_result.append(max_value_coordinates[1])
            point_wise_result.append(max_value_coordinates[0])
            result.append(point_wise_result)
        writer.writerows(result)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Base settings
    parser.add_argument('--img_info', type=str, default='/data2/serie/Dataset/XCAT_CT_info_updated.csv')
    parser.add_argument('--projM_path', type=str, default='/data2/serie/Dataset/ProjMat_revised/')
    parser.add_argument('--height_2d', type=int, default=480)
    parser.add_argument('--width_2d', type=int, default=620)
    parser.add_argument('--xy_voxel_3d', type=int, default=800)
    ## Edit here
    parser.add_argument('--tif_path', type=str, default='/data2/serie/Head_Initial_predictions/HTC_multires_1.75/train_output_heatmaps_tiff/')
    parser.add_argument('--result_output_path', type=str, default='/data2/serie/Head_Initial_predictions/HTC_multires_1.75/train_3d_reference_coordinates/')
    parser.add_argument('--condition', type=str, default='train', choices=['test', 'train'])
    config = parser.parse_args()
    main(config)