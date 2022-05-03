import torch
import torchvision
import torchvision.transforms as transforms
# %%
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# %%
print(trainloader.shape)
# %%
import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# %%
def select_cat_and_dog(trainloader):
    dataiter = iter(trainloader)
    img = torch.randn(1, 3, 32, 32)
    l = torch.tensor(0).reshape(1)
    cnt=0
    while(1):
        try:
            images, labels = dataiter.next()
            for i in range(1):
                if (labels[i] == 3):
                    img = torch.vstack((img, torch.unsqueeze(images[i], 0)))
                    l = torch.cat((l, torch.tensor(0).reshape(1)))
                elif (labels[i] == 5):
                    img = torch.vstack((img, torch.unsqueeze(images[i], 0)))
                    l = torch.cat((l, torch.tensor(1).reshape(1)))
        except:
            break
    return img, l
imgs, img_labels = select_cat_and_dog(trainloader)
#%%
print(imgs.shape)
print(img_labels.shape)
#%%
from torch.utils.data import  TensorDataset, DataLoader
ds = TensorDataset(imgs, img_labels)
trainloader_se = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                        shuffle=True, num_workers=0)
#%%
dataiter_se = iter(trainloader_se)
images_se, labels_se = dataiter_se.next()
print(images_se.shape)
print(labels_se.shape)
imshow(torchvision.utils.make_grid(images_se))