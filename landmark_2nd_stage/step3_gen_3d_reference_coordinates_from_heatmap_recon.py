import numpy as np
import os
import argparse
from PIL import Image,ImageSequence
import pandas as pd
import cv2
import csv
import natsort
'''
This code converts 3D reference landmark coordinates
to 2D forward projected landmarks.
'''
def main(config):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path) 
    
    if config.condition == 'test':
        patient_li = ['female_3_head', 'female_4_head', 'female_6_head', 'female_7_head', 'female_10_head', 'male_3_head', 'male_4_head', 'male_7_head', 'male_12_head']
    elif config.condition == 'train':
        patient_li = ['female_1_head', 'female_5_head', 'female_8_head', 'female_9_head', 'female_11_head', 'female_12_head', 'female_13_head', 'female_14_head', 'female_15_head',
                       'male_1_head', 'male_2_head', 'male_5_head', 'male_6_head', 'male_8_head', 'male_9_head', 'male_10_head', 'male_11_head', 'male_13_head']

    for i, patient in enumerate(patient_li):
        print("Calculating {}...".format(patient))
        if patient in ['female_4_head', 'female_5_head', 'female_6_head', 'female_7_head', 'female_8_head', 'female_10_head', 'male_1_head', 'male_2_head', 'male_3_head', 'male_5_head', 'male_6_head', 'male_9_head', 'male_10_head']:
            vol_pat = 250
        elif patient in ['female_1_head', 'female_3_head']:
            vol_pat = 300
        elif patient in ['female_11_head', 'female_12_head', 'female_13_head', 'female_14_head', 'female_15_head', 'male_4_head', 'male_12_head']:
            vol_pat = 240
        elif patient in ['male_7_head', 'male_11_head', 'male_13_head']:
            vol_pat = 260
        elif patient in ['female_9_head']:
            vol_pat = 230
        elif patient in ['male_8_head',]:
            vol_pat = 220

        img_path = config.img_path + 'Forward Projection of {}.tif'.format(patient)
        proj_path = config.projM_path + '{}.txt'.format(patient)
        ref_3d_csv = config.input_csv + '{}_head.csv'.format(patient.split('_head')[0])
        label_output_folder = config.output_path + '/{}/'.format(patient.split('_head')[0])

        if not os.path.exists(label_output_folder):
            os.makedirs(label_output_folder)
        in_img = Image.open(img_path)
        im = ImageSequence.Iterator(in_img)
        for i, im in enumerate(ImageSequence.Iterator(in_img)):
            output_file_name = label_output_folder + patient + '_{}.txt'.format(i)
            f_w = open(output_file_name,'w')
            im = Image.fromarray(im/np.amax(im) * 255)
            im = np.array(im.convert('L'))
            landmark_3d = pd.read_csv(ref_3d_csv, sep=",", header=None)
            proj = np.loadtxt(proj_path, delimiter=' ')
            p_i = proj[i]
            im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
            proj_i = np.array([[p_i[0],p_i[1],p_i[2],p_i[3]],
                    [p_i[4],p_i[5],p_i[6],p_i[7]],
                    [p_i[8],p_i[9],p_i[10],p_i[11]],
                    ])                
            point_2d = []
            for k in range(len(landmark_3d.index)):
                x = float(((landmark_3d.iloc[k][0])-400)*0.5)
                y = float(((landmark_3d.iloc[k][1])-400)*0.5)
                z = float(((landmark_3d.iloc[k][2])-((vol_pat/2)-0.5)))
                point_2d = np.matmul(proj_i,np.array([x,y,z,1]))
                point_2d = (point_2d/point_2d[2])
                u = point_2d[0]
                v = point_2d[1]
                point_2d = np.append(point_2d, [u,v])
                f_w.writelines(str(u)+','+str(v))
                f_w.writelines('\n')
            f_w.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Base settings
    parser.add_argument('--img_path', type=str, default='E:/Dataset/Head_Phantom/Forward_Projection_wo_motion_updated/')
    parser.add_argument('--projM_path', type=str, default='E:/Dataset/Head_Phantom/ProjMat_revised/')
    ## Edit here
    parser.add_argument('--input_csv', type=str, default='E:/Heatmap_Refinement/Head_initial_predictions/HTC_v1_multires_1.75/train_3d_reference_coordinates/')
    parser.add_argument('--output_path', type=str, default='E:/Heatmap_Refinement/Head_initial_predictions/HTC_v1_multires_1.75/train_2d_reference_coordinates/')
    parser.add_argument('--condition', type=str, default='train', choices=['test', 'train'])
    config = parser.parse_args()
    main(config)