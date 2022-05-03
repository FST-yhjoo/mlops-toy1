#%%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os 
from pytorch_lightning import LightningModule, Trainer, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision import datasets
import PIL.Image as Image
from sklearn.model_selection import train_test_split
from config.config import Config

#%%
def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = torch.utils.data.Subset(dataset, train_idx)
    datasets['val'] = torch.utils.data.Subset(dataset, val_idx)
    return datasets
    
class DataModule(LightningDataModule):
    def __init__(self, data_dir='./', batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.transform = transforms.ToTensor()
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        )
        self.transform = transforms.Compose([
            # transforms.RandomCrop((56,56)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            transforms.Resize((112, 112)),
            normalize
        ])
        self.config = Config()
        paths = self.config.get_data_path()    
        self.train_data_path  = paths[0]
        self.test_data_path   = paths[1]
        self.sample_data_path = paths[2]
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = datasets.ImageFolder(self.train_data_path, transform = self.transform)
            train_size = int(0.8 * len(train_dataset))
            test_size = len(train_dataset) - train_size
            self.train_data, self.valid_data = torch.utils.data.random_split(train_dataset, [train_size, test_size])

        if stage == 'test' or stage is None:
            self.test_data = datasets.ImageFolder(self.test_data_path, transform = self.transform)
        
    def train_dataloader(self):
        '''returns training dataloader'''
        img_train = torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True)
        return img_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        img_valid = torch.utils.data.DataLoader(
            self.valid_data, 
            batch_size=self.batch_size,
            num_workers=0, 
            pin_memory=True)
        return img_valid

    def test_dataloader(self):
        '''returns test dataloader'''
        img_test = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=self.batch_size,
            num_workers=8, pin_memory=True)
        return img_test

    def return_sample(self):
        path = self.sample_data_path
        img_list = os.listdir(path)
        img_name = img_list[10]
        img = Image.open(path + os.sep + img_name)
        return img