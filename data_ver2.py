# %%
import os 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL.Image as Image

class ImageTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.RandomCrop((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            # transforms.Resize((32, 32))
        ])

    def __call__(self, img):
        return self.data_transform(img)

class DataModule(data.Dataset):

    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.file_list = self.make_file_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index] # 데이터셋에서 파일 하나를 특정
        img = Image.open(self.path + os.sep + img_path)
        
        img_transformed = self.transform(img)
        img_transformed = img_transformed[None,:, :, :]
        
        if img_path[0:1] == "c":
            label = torch.tensor(0).reshape(1)
        elif img_path[0:1] == "d":
            label = torch.tensor(1).reshape(1)

        return img_transformed, label
    
    def make_file_list(self):
        file_list = os.listdir(self.path)
        return file_list


# %%
