import numpy as np
import os
from PIL import Image,ImageSequence
import pandas as pd
import cv2
import csv

#patient_li = ['N1', 'N2', 'N4', 'N6', 'N7', 'N9', 'N10', 'N11', 'N12', 'N13', 'N15', 'N16', 'N17', 'N21', 'N23', 'N24', 'N26', 'N27']
# patient_li = ['female_1_head','female_3_head','female_4_head','female_5_head','female_6_head','female_7_head','female_8_head',
#             'female_9_head','female_10_head','female_11_head','female_12_head','female_13_head','female_14_head','female_15_head',
#             'male_1_head','male_2_head','male_3_head','male_4_head','male_5_head','male_6_head','male_7_head',
#             'male_8_head','male_9_head','male_10_head','male_11_head','male_12_head','male_13_head']

patient_li = ['female_4_head','male_7_head','female_3_head', 'male_3_head', 'female_10_head', 'female_7_head', 'male_12_head', 'male_4_head', 'female_6_head']

# data = pd.read_csv("E:/Dataset/Brain_CT_Hemorrhage_Dataset/Normal_info_n.csv", header=True)
# print(data)
# print(data.iloc[0])

def search_csv(filename, search_value):
    return [row for row in csv.reader(open(filename, 'r', newline='')) if row[0] == search_value]

for i, p in enumerate(patient_li):

    patient = p
    #img_path = 'E:/Dataset/Brain_CT_Hemorrhage_Dataset/Forward_projection/{}.tif'.format(patient)
    #proj_path = 'E:/Dataset/Brain_CT_Hemorrhage_Dataset/Projection_Matrices_revised/{}.txt'.format(patient)
    #label_3d_path = 'E:/Dataset/Brain_CT_Hemorrhage_Dataset/3d_csv_updated/{}.csv'.format(patient)
    #output_folder2 = 'E:/Dataset/Brain_CT_Hemorrhage_Dataset/6points_n/'
    #output_folder = 'E:/Dataset/Brain_CT_Hemorrhage_Dataset/6points_n/2D_Labels_gt/'
    #label_output_folder = 'E:/Dataset/Brain_CT_Hemorrhage_Dataset/6points_n/2D_Labels/'
    #output_proj_2d = 'E:/Dataset/Brain_CT_Hemorrhage_Dataset/6points_n/2D_Labels/'+patient+'.txt'
    #ANGLE = 360
    img_path = 'E:/Dataset/Head_Phantom/Forward_Projection_wo_motion_updated/Forward Projection of {}.tif'.format(patient)
    proj_path = 'E:/Dataset/Head_Phantom/ProjMat_revised/{}.txt'.format(patient)
    # label_3d_path = 'E:/Dataset/Head_Phantom/csv_label/11points_avg/{}.csv'.format(patient)
    label_3d_path = 'E:/Heatmap_Refinement/coord2hm_generation/female_3/results_csv/female_3_heatmap_refinement.csv'
    output_folder2 = 'E:/Heatmap_Refinement/coord2hm_generation/female_3/'
    # output_folder = 'E:/Heatmap_Refinement/coord2hm_generation/female_3/2D_Lables_/'
    label_output_folder = 'E:/Heatmap_Refinement/coord2hm_generation/female_3/2D_Labels/'
    output_proj_2d = 'E:/Heatmap_Refinement/coord2hm_generation/female_3/2D_Labels/'+patient+'.txt'

    search_v = patient
    #p_info_li = search_csv('E:/Dataset/Brain_CT_Hemorrhage_Dataset/Normal_info_n.csv', search_v)
    #print("p_info_li: ", p_info_li)

    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    if not os.path.exists(label_output_folder):
        os.makedirs(label_output_folder)
    f_w = open(output_proj_2d,'w')
        
    in_img = Image.open(img_path)
    im = ImageSequence.Iterator(in_img)

    for i, im in enumerate(ImageSequence.Iterator(in_img)):
        #im = np.array(im[i])
        im = Image.fromarray(im/np.amax(im) * 255)
        im = np.array(im.convert('L'))
        #print('im_size:',im.shape)
        landmark_3d = pd.read_csv(label_3d_path, sep=",", header=None)
        proj = np.loadtxt(proj_path, delimiter=' ')
        p_i = proj[i]
        im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        #im = cv2.flip(im, 1)
        proj_i = np.array([[p_i[0],p_i[1],p_i[2],p_i[3]],
                [p_i[4],p_i[5],p_i[6],p_i[7]],
                [p_i[8],p_i[9],p_i[10],p_i[11]],
                ])                
        point_2d = []
        for k in range(len(landmark_3d.index)):
            #z_volume = 
            if patient == 'N1' or patient == 'N2':
                z_spacing = 0.5
            else:
                z_spacing = 1
            x = float(((landmark_3d.iloc[k][0])-400)*0.5)
            y = float(((landmark_3d.iloc[k][1])-400)*0.5)
            z = float(((landmark_3d.iloc[k][2])-149.5))
            #print(x, y, z)
            point_2d = np.matmul(proj_i,np.array([x,y,z,1]))
            #print(point_2d)
            point_2d = (point_2d/point_2d[2])
            u = point_2d[0]
            v = point_2d[1]
            #print(u, v)
            point_2d = np.append(point_2d, [u,v])
            f_w.writelines(str(u)+','+str(v))
            f_w.writelines('\n')
            image = cv2.circle(im, (int(u),int(v)), radius=1, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(os.path.join(output_folder2,patient+'_'+str(i)+'.png'),im)

    f_w.close()