#%%
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
#%%
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
#%%
# net.to(device)

for epoch in range(2):   # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    for i, data in enumerate(trainloader_se, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
# %%
dataiter_se = iter(trainloader_se)
images_se, labels_se = dataiter_se.next()
# print(images_se.shape)
# print(labels_se.shape)
# imshow(torchvision.utils.make_grid(images_se))

outputs = net(images_se[0])
_, predicted = torch.max(outputs, 1)
imshow(torchvision.utils.make_grid(images_se))
print(predicted)
# %%
print(predicted)
# %%
