import torch
from data_ver2 import DataModule
from data_ver2 import ImageTransform
from model_ver1 import vanillaNet
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os 

def main(iter=2):
    
    img_data = DataModule(
        path=os.getcwd()+"/data/train/0",
        transform=ImageTransform()
    )
    img_model = vanillaNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(img_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(iter):   

        running_loss = 0.0
        for img, label in img_data:

            optimizer.zero_grad()

            outputs = img_model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    torch.jit.save(img_model.state_dict(), "/models")
    print('Finished Training')
if __name__ == "__main__":
    main()
