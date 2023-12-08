# Created: 210313 14:02
# Last edited: 210421 14:02 

import os
import shutil

def mutil_filter(input_label_dir,output_label_dir):
    # 筛除多行的label.txt
    # input_label_dir = './dataset1/train/labels_filtered/'
    # output_label_dir = './dataset1/train/labels_filtered2/'
    label_name_list = os.listdir(input_label_dir)
    if not os.path.exists(output_label_dir):
        os.mkdir(output_label_dir)

    for label_name in label_name_list:
        label_file_name = input_label_dir + label_name
        with open(label_file_name, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
        # print(lines)
        if len(lines) > 1:
            # print(lines[1])
            # 注意这里他有一个把多行的label拷贝到其他文件夹里面,不知道用处是什么，所以先注释掉，因为出现的是需要创建的路径
            # shutil.copy(label_file_name, './dataset2/math/mult-line_label/' + label_name)
            print(f'mutli: {label_name}')
            continue
        shutil.copy(label_file_name, output_label_dir + label_name)
    # 筛除多行的label.txt end

def error_filter(input_label_dir,output_label_dir):
    # 筛除error mathpix
    # input_label_dir = './dataset1/train/labels_filtered2/'
    # output_label_dir = './dataset1/train/labels_filtered3/'
    label_name_list = os.listdir(input_label_dir)
    if not os.path.exists(output_label_dir):
        os.mkdir(output_label_dir)

    for label_name in label_name_list:
        print(label_name)
        label_file_name = input_label_dir + label_name
        with open(label_file_name, 'r', encoding='utf-8') as f1:
            content = f1.read()
        if 'error mathpix' in content:
            print(f'error: {label_name}')
            continue
        shutil.copy(label_file_name, output_label_dir + label_name)
    # 筛除error mathpix end