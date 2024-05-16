import numpy as np
import cv2
import os
import natsort
from PIL import Image
import argparse

'''
This code converts output coordinate predictions to 2D heatmap tiff file.
'''

# test
# patient_li = ['female_3', 'female_4', 'female_6', 'female_7', 'female_10', 'male_3', 'male_4', 'male_7', 'male_12']
# train
# patient_li = ['female_1', 'female_5', 'female_8', 'female_9', 'female_11', 'female_12', 'female_13', 'female_14', 'female_15', 
#              'male_1', 'male_2', 'male_5', 'male_6', 'male_8', 'male_9', 'male_10', 'male_11', 'male_13']

# CQ500 test
# patient_li = ['CQ500CT127', 'CQ500CT142', 'CQ500CT180', 'CQ500CT231', 'CQ500CT25', 'CQ500CT349']
# patient_li = ['CQ500CT12', 'CQ500CT140', 'CQ500CT148', 'CQ500CT191', 'CQ500CT20', 'CQ500CT227', 'CQ500CT235', 'CQ500CT269', 'CQ500CT318', 'CQ500CT336', 'CQ500CT355', 'CQ500CT35', 'CQ500CT375', 'CQ500CT463', 'CQ500CT81', 'CQ500CT9']



def generate_gaussian_heatmap(center, sigma, width, height):
    x, y = np.meshgrid(np.arange(width), np.arange(height)) # Create a grid of points to calculate Gaussian values
    x0, y0 = center # Calculate the Gaussian values
    heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    heatmap = heatmap / heatmap.max()   # Normalize the heatmap to have values between 0 and 1
    return heatmap

def save_as_tif(patient, j, images, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_filename = output_path + '/point_{}.tif'.format(j)
    print("output_filename", output_filename)
    images[0].save(output_filename, save_all=True, append_images=images[1:], compression='tiff_lzw')
    #print(f"Merged {len(images)} PNG images into {output_filename}")

def main(config):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    # Per patients
    for patient in patient_li:
        print("Saving from ", patient, "...")
        # Loading coordinates
        
        open_path = config.coord_path + patient + '_coordinate_output/'
        print(open_path)
        angle_files = natsort.natsorted(os.listdir(open_path))
        point0, point1, point2, point3, point4, point5, point6, point7, point8, point9, point10 = [],[],[],[],[],[],[],[],[],[],[]
        
        for f in angle_files:
            # Loading coordinates from text files, generating Gaussian heatmaps
            with open(open_path+f, "r") as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    x, y = int(float(line.split(',')[0])), int(float(line.split(',')[1]))
                        
                    heatmap = generate_gaussian_heatmap((x, y), config.sigma, config.width, config.height)
                    if i==0:
                        point0.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==1:
                        point1.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==2:
                        point2.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==3:
                        point3.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==4:
                        point4.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==5:
                        point5.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==6:
                        point6.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==7:
                        point7.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==8:
                        point8.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    elif i==9:
                        point9.append(Image.fromarray((heatmap * 255).astype(np.uint8)))
                    # elif i==10:
                    #    point10.append(Image.fromarray((heatmap * 255).astype(np.uint8)))

        # Save the list of images as a TIF
        save_as_tif(patient, 0, point0, config.output_path + patient)
        save_as_tif(patient, 1, point1, config.output_path +patient)
        save_as_tif(patient, 2, point2, config.output_path+patient)
        save_as_tif(patient, 3, point3, config.output_path+patient)
        save_as_tif(patient, 4, point4, config.output_path+patient)
        save_as_tif(patient, 5, point5, config.output_path+patient)
        save_as_tif(patient, 6, point6, config.output_path+patient)
        save_as_tif(patient, 7, point7, config.output_path+patient)
        save_as_tif(patient, 8, point8, config.output_path+patient)
        save_as_tif(patient, 9, point9, config.output_path+patient)
        # save_as_tif(patient, 10, point10, config.output_path+patient)
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coord_path', type=str, default = 'E:/Heatmap_Refinement/CQ500_initial_predictions/HTC_v1_multires_CQ500/train_output_heatmaps/')
    parser.add_argument('--output_path', type=str, default = 'E:/Heatmap_Refinement/CQ500_initial_predictions/HTC_v1_multires_CQ500/train_2d_output_heatmaps/')
    parser.add_argument('--sigma', type=int, default = 2)
    parser.add_argument('--height', type=int, default = 600)
    parser.add_argument('--width', type=int, default = 800)
    parser.add_argument('--nLandmarks', type=int, default = 10)
    config = parser.parse_args()
    main(config)