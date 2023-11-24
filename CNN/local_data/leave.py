import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from PIL import Image


data_root_dir = 'C:\\Users\\cheng\\Desktop\\d2l\\data\\classify-leaves'
data_csv_path = os.path.join(data_root_dir, './train.csv')

class Dataset(object):
    def __init__(self, data_frame, root_dir, transform=None) -> None:
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        imgpath = os.path.join(self.root_dir, self.data_frame.iloc[idx,0])
        label = self.data_frame.iloc[idx,2]
        img = Image.open(imgpath).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label


def load_csv(path):
    df = pd.read_csv(path)
    list_type = df['label'].unique()
    dic = {}
    for i,t in enumerate(list_type):
        dic[t] = i
    df['num_label'] = df['label'].map(dic)
    return shuffle(df)

def load_data(transform):
    data_df = load_csv(data_csv_path)
    train_dataset = Dataset(data_df.iloc[:14000], data_root_dir, transform)
    test_dataset = Dataset(data_df.iloc[14000:], data_root_dir, transform)

    return train_dataset, test_dataset