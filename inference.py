import torch
from model_ver2 import LitResnet
from data_ver3 import DataModule
import PIL.Image as Image
import os
import numpy as np

class Predictor:

    def __init__(self, model_path):
        self.model_path = model_path
        model_list =  os.listdir(self.model_path)
        model_name = model_list[-2]
        self.model = LitResnet.load_from_checkpoint(checkpoint_path =self.model_path+model_name)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.lables = ["0", "1"]

    def predict(self, img):
        processed = self.processor.transform(img)
        processed = processed.unsqueeze(0)
        logits = self.model(processed)
        
        scores = self.softmax(logits).tolist()[0]
        if scores[0] > scores[1]:
            output = "cat"
        else:
            output = "dog"
        score= np.max(scores)
        return {"result": output, "score":score}

if __name__ == "__main__":
    predictor = Predictor("models/")
    img_datamodule = DataModule()
    img_sample = img_datamodule.return_sample()
    with torch.no_grad():
        result = predictor.predict(img_sample)
        print(result)