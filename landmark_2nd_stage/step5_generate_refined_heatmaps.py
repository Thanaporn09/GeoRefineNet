import os
import numpy as np
import cv2
from scipy.stats import multivariate_normal
import argparse

def main(config):
    heatmap_width = config.image_width
    heatmap_height = config.image_height
    input_directory = config.input_directory
    output_directory = config.output_directory
    sigma = config.sigma

    people_directories = [f for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 각 사람의 디렉토리에 대해 작업 수행
    for person_directory in people_directories:
        person_input_directory = os.path.join(input_directory, person_directory)
        person_output_directory = os.path.join(output_directory, person_directory)

        txt_files = [f for f in os.listdir(person_input_directory) if f.endswith(".txt")]

        #if not os.path.exists(person_output_directory):
        #    os.makedirs(person_output_directory)

        for txt_file in txt_files:
            file_path = os.path.join(person_input_directory, txt_file)
            with open(file_path, "r") as f:
                lines = f.readlines()
            heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
            for line in lines:
                x, y = map(float, line.strip().split(","))
                x = int(x)
                y = int(y)
                mean = [x, y]
                cov = [[sigma, 0], [0, sigma]]
                X, Y = np.meshgrid(np.arange(heatmap_width), np.arange(heatmap_height))
                pos = np.dstack((X, Y))
                heatmap += multivariate_normal.pdf(pos, mean, cov)

            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255

            output_file = os.path.splitext(txt_file)[0] + ".png"
            output_path = os.path.join(output_directory, output_file)
            cv2.imwrite(output_path, heatmap.astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Base settings
    parser.add_argument('--image_height', type=int, default=480)
    parser.add_argument('--image_width', type=int, default=620)
    parser.add_argument('--sigma', type=int, default=2)
    ## Edit here
    # parser.add_argument('--input_directory', type=str, default='E:/Heatmap_Refinement/CQ500_initial_predictions/HRNet_w48_baseline/train_2d_reference_coordinates/')
    # parser.add_argument('--output_directory', type=str, default="E:/Heatmap_Refinement/CQ500_initial_predictions/HRNet_w48_baseline/train_2d_reference_heatmaps/")
    parser.add_argument('--input_directory', type=str, default='E:/Heatmap_Refinement/Head_initial_predictions/HTC_v1_multires_1.75/test_2d_reference_coordinates/')
    parser.add_argument('--output_directory', type=str, default="E:/Heatmap_Refinement/Head_initial_predictions/HTC_v1_multires_1.75/test_2d_reference_heatmaps/")
    
    config = parser.parse_args()
    main(config)