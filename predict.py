import os
import json
import  pandas as pd
import torch
from torchvision import transforms

from model import resnet34
from Torch_Train.MyDataSet import  My_Dataset as md

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #num to class file
    json_path=r"E:\DataVisualization\Torch_Train\MyDataSet\num_to_class.json"
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test":transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    batch_size = 16
    nw = 0  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    test_data_path=r"E:\DataVisualization\Torch_Train\data\test.csv"
    test_dataset=md.My_Dataset(data_path=test_data_path,transform=data_transform["test"],train=False)
    test_loader=torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=nw)
    net = resnet34(num_classes=176).to(device)

    model_weight_path = "./resNet34_classify_leaves_50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    net.eval()
    pred=[]
    with torch.no_grad():
        for images in test_loader:
            outputs=net.forward(images.to(device))
            pred.append(torch.argmax(outputs,dim=1).cpu().numpy())

    with open(json_path,"r") as f:
        num_to_class=json.load(f)
    result=pd.read_csv(test_data_path)
    index=len(result)
    predicted_labels={}
    preidcted=[]
    for i in pred:
        for j in i:
            preidcted.append(num_to_class[str(j)])
    predicted_labels["label"]=preidcted
    result["label"]=predicted_labels["label"]
    result.to_csv("result.csv")

if __name__ == "__main__":
    main()



