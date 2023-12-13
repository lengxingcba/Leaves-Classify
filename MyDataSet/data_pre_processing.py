import  pandas as pd
import  numpy as np
import os
import json
import random

import shutil
# root_path="E:\DataVisualization\Torch_Train\data"
# image_path=os.path.join(root_path,"images")
# train=os.path.join(root_path,"train.csv")
# train=pd.read_csv(train)
# labels=train.drop_duplicates(subset="label",keep="first")  #标签去重获取所有类别
# labels=list(labels["label"])    #获取所有类别转成list格式
# num_class=len(labels)           #获取分类数量
# class_indices=dict((k,v) for v,k in enumerate(labels,0))
# print(class_indices)
# json_str=json.dumps(dict((key,val) for key,val in class_indices.items()),indent=16)
# with open("class_to_num.json","w") as json_file:
#     json_file.write(json_str)

def make_trainvaldata(imagepath:str,rate):
    image_name=os.listdir(imagepath)
    list_image=[os.path.join(imagepath,i) for i in image_name]
    list_val=random.sample(list_image,int(rate*len(list_image)))
    list_train=[]
    for image in list_image:
        if image not in list_val:
            list_train.append(image)
    with open(r"E:\DataVisualization\Torch_Train\data\Fire_image\my_train_data.txt","w") as f:
        f.write("\n".join(list_train))
        f.close()
    with open(r"E:\DataVisualization\Torch_Train\data\Fire_image\my_val_data.txt","w") as f:
        f.write("\n".join(list_val))
        f.close()
    #将图片分到train文件夹和val文件夹下
    path=os.path.abspath(os.path.join(imagepath, ".."))
    path_train = os.path.join(path, "train")
    path_val = os.path.join(path, "val")
    if not os.path.exists(os.path.join(path, "train")):
        os.makedirs(os.path.join(path,"train"))
        os.makedirs(os.path.join(path,"val"))
    for p in list_train:
        shutil.move(p,os.path.join(path_train,p.split("\\")[6]))
    for p in list_val:
        shutil.make_archive(p,os.path.join(path_val,p.split("\\")[6]))
# make_trainvaldata(imagepath=r"E:\DataVisualization\Torch_Train\data\Fire_image\data",rate=0.3)


def make_label(datapath:str,annotationpath:str):
    abs_path=os.path.abspath(os.path.join(datapath, ".."))
    with open(datapath,"r") as f:
        lines=f.readlines()
    lines=[l.strip() for l in lines]
    lines=[l.split("\\")[6] for l in lines]
    lines=[l[:-4] for l in lines]
    print(lines,len(lines))
    if not os.path.exists(os.path.join(abs_path,"labels")):
        os.makedirs(os.path.join(abs_path,"labels"))
        label_path=os.path.join(abs_path,"labels")
    for l in lines:
        if os.path.exists(os.path.join(annotationpath,l+".txt")):
            shutil.copy(os.path.join(annotationpath,l+".txt"),os.path.join(label_path,l+".txt"))
        else:
            print("path:{} does not exist".format(os.path.join(annotationpath,l+".txt")))
# make_label(datapath=r"E:\DataVisualization\Torch_Train\data\Fire_image\my_val_data.txt",annotationpath=r"E:\DataVisualization\Torch_Train\data\ForestFireImage\Annotations")