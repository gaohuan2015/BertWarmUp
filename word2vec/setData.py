import numpy as np
from torch.utils.data import Dataset

class setData(Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, data, label):
        self.Data = data
        self.Label = label
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return {'x_data': data, 'y_target': label}