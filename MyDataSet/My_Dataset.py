import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import  torch
import pandas as pd
import  json


class My_Dataset(Dataset):
    def __init__(self,data_path,transform=None,train=False):
        self.data_path=data_path
        self.transform=transform
        self.train=train
        data=pd.read_csv(self.data_path,header=None)
        self.root_path="E:/DataVisualization/Torch_Train/data/"
        self.image_path=np.asarray(data.iloc[1:,0])
        if self.train:
            self.image_class=np.asarray(data.iloc[1:,1])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        path=os.path.join(self.root_path, self.image_path[item])
        path_class2num="E:\DataVisualization\Torch_Train\MyDataSet\class_to_num.json"
        #读取标签到数字标签的json文件并返回字典
        f = open(path_class2num, "r")
        content = f.read()
        num_to_class = json.loads(content)

        if os.path.exists(path):
            img=Image.open(path)
            if img.mode !="RGB":
                raise ValueError("image {} is not RBG mode".format(os.path.join(self.root_path,self.image_path[item])))
            if self.train:
                label=self.image_class[item]
                label=num_to_class[label]
            if self.transform is not None:
                img = self.transform(img)
            if self.train:
                return img, label
            else:
                return img
        else:
            print("path {} do not exist".format(path))

    # @staticmethod
    # def collate_fn(batch):
    #     images, labels = tuple(zip(*batch))
    #
    #     images = torch.stack(images, dim=0)
    #     labels = torch.as_tensor(labels)
    #     return images, labels