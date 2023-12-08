import os
import shutil

def extract_image_by_id(label_dir,image_dir,output_dir):
    # label_dir = './dataset1/train/labels_filtered3/'
    # # label_dir = './data/math_210421/formula_labels_210421_no_chinese/'
    # image_dir = './dataset1/train/images/'

    # output_dir ='./dataset1/train/images_extract/'
    
    # 根据label路径获得所有label的name(id)
    label_name_list = os.listdir(label_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(len(label_name_list)):
        label_name_list[i] = label_name_list[i][:-4]

    # print(label_list)
    # 根据image路径获得所有image的name(id)
    image_name_list = os.listdir(image_dir)
    # 把存在对应id的image拷贝
    for image_name in image_name_list:
        if image_name[:-4] in label_name_list:
            print(image_name)
            shutil.copy(image_dir + image_name, output_dir + image_name)