import pandas as pd
import  numpy as np
import  random as r
import  json
import pickle
path=r"E:\DataVisualization\Torch_Train\data\train.csv"
def make_test_data(data_path,k): #k为采样概率
    total_data=pd.read_csv(data_path)
    images=np.asarray(total_data.iloc[1:,0])
    label=np.asarray(total_data.iloc[1:,1])
    d=dict(zip(images,label))
    images=r.sample(d.getkeys(),len(d)*k)
    with open("voilddata.json","wb") as fp:
        pickle.dump(images,fp)

#make_test_data(path,k=0.3)
def read_json(json_path:str):
    with open(json_path,"r") as f:
        a=json.load(f)
    return a

def save_as_json(filename,savepath):
    with open(savepath,"w") as f:
        json.dump(filename,f)

def num_to_class(datapath):
    data=read_json(datapath)
    num_to_class={v:k for k,v in data.items()}
    save_as_json(num_to_class,"num_to_class.json")


testdata=r"./class_to_num.json"
num_to_class(testdata)









