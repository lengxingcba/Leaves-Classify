import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns

import MyDataSet.My_Dataset as md

# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False
# resnet34模型
def res_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

#读取数据
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]),
    "val": transforms.Compose([transforms.Resize(224),
                               #transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])}

batch_size=16
data_train=md.My_Dataset(data_path="E:/DataVisualization/Torch_Train/data/train.csv",transform=data_transform["train"],train=True)
data_valid=md.My_Dataset(data_path=r"E:\DataVisualization\Torch_Train\data\valid.csv",transform=data_transform["val"],train=True)
train_loader=DataLoader(data_train,shuffle=True,batch_size=batch_size)
valid_loader=DataLoader(data_valid,shuffle=False,batch_size=batch_size)
