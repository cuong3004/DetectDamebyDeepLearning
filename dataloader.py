from glob import glob
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import torch
from PIL import Image

batch_size = 32
num_workers = 2

pl.seed_everything(1234)

labels_dir = {
    "Acne" : 0,
    "Eczema" : 1,
    "Nail_Fungus" : 2,
    "Psoriasis" : 3,
    "Normal": 4
}

key_list = list(labels_dir.keys())
val_list = list(labels_dir.values())

path_img = glob("tonthuongda/train/*/*")
path_train, path_valid = train_test_split(path_img ,shuffle=True,test_size=0.1)

path_test = glob("tonthuongda/test/*/*")

def generate_dataframe(path_file, name_df):
    name = [i.split("/")[-2] for i in path_file]
    index = [labels_dir[i] for i in name]

    df = pd.DataFrame()
    df["Path Image"] = path_file 
    df["Name Disease"] = name 
    df["Index Value"] = index

    df.to_csv(name_df, index=False)
    return df

train = generate_dataframe(path_train, "train.csv") 
valid = generate_dataframe(path_valid, "valid.csv") 
test = generate_dataframe(path_test, "test.csv") 

def preprocess(img):
    img_new = ((img - 127.5)/127.5)
    return img_new


transform = A.Compose([
    A.Blur(),
    A.Cutout(),
    A.ISONoise(p=0.7),
    A.RandomBrightnessContrast(),
    A.Transpose(),
    A.VerticalFlip(),
    A.HorizontalFlip(),
    A.Resize(224,224),
    A.Normalize(mean=127.5, std=127.5, max_pixel_value=1.0),
    ToTensorV2(),

])
transform_valid = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=127.5, std=127.5, max_pixel_value=1.0),
    ToTensorV2(),
])
transform_test = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=127.5, std=127.5, max_pixel_value=1.0),
    ToTensorV2(),
])

class CustomData(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __getitem__(self, idx):
        id_data = self.data_frame.values[idx]
        img = np.array(Image.open(id_data[0]).convert("RGB"))
        label = id_data[-1]
        label = torch.tensor(label)

        if self.transform:
            tranformed = self.transform(image=img)
            img = tranformed['image']

        assert img.dtype == torch.float32
        return img, label

    def __len__(self):
        return len(self.data_frame)

def get_class_distribution(dataset_obj):
    count_array = np.array([len(np.where(dataset_obj == t)[0]) for t in np.unique(dataset_obj)])
    return count_array

target_list = train['Index Value'].values

class_count = np.array([i for i in get_class_distribution(train['Index Value'].values)])
weight = 1. / class_count
samples_weight = np.array([weight[t] for t in target_list])
samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()

weighted_sampler = WeightedRandomSampler ( 
    weights = samples_weigth, 
    num_samples = len (samples_weigth), 
)

data_train = CustomData(data_frame=train, transform=transform)
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False, sampler = weighted_sampler)

data_vaild = CustomData(data_frame=valid, transform=transform_valid)
valid_loader = DataLoader(data_vaild, batch_size=batch_size, shuffle=False)

data_test = CustomData(data_frame=test, transform=transform_test)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    
    print(len(train_loader))
    print(len(valid_loader))
    print(len(test_loader))