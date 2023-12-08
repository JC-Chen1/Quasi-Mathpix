import os
import pickle
from fmm import tokenizer,FMM_func
from data_filter import mutil_filter,error_filter
from extract_image_according_to_label_list import extract_image_by_id
from pad_image import set_pad_img

def dataset(input_dir,output_dir):
    # 三次过滤后的label路径
    # input_dir='./dataset1/train/labels_filtered3/'
    # output_dir='./dataset1/train/anno_pure_train.pkl'
    label_name_list = os.listdir(input_dir)

    annotation=[]
    for label_name in label_name_list:
        with open(input_dir + label_name,'r',encoding='utf-8') as f:
            train_label_name = label_name[:-4]
            line = f.read()
            annotation.append({"id":int(train_label_name),"annotation":line})
            
    with open(output_dir, 'wb') as f:
        pickle.dump(annotation, f, 0)

if __name__ == '__main__':
    # 在这里修改对应的路径
    input_dir='./dataset1/train/labels/'
    input_raw_image='./dataset1/train/images/'
    output_dir1='./dataset1/train/label_filter1/'
    output_dir2='./dataset1/train/label_filter2/'
    output_dir3='./dataset1/train/label_filter3/'
    output_image='./dataset1/train/extract_image/'
    output_pad_image='./dataset1/train/pad_image/'
    output_pkl='./dataset1/train/anno_train.pkl'
    # 首先分词
    tokenizer(input_dir,output_dir1)
    # 进行多行过滤和错误过滤
    mutil_filter(output_dir1,output_dir2)
    error_filter(output_dir2,output_dir3)
    #根据标签提取对应的image
    extract_image_by_id(output_dir3,input_raw_image,output_image)
    #给img加上合适的pad
    set_pad_image(output_image,output_pad_image)
    # 获取label对应的pkl
    dataset(output_dir3,output_pkl)

# output_image='./dataset1/train/extract_image/'
# output_pad_image='./dataset1/train/pad_image/'
# set_pad_img(output_image,output_pad_image)





