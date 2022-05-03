#%%
import torch
from model_ver2 import LitResnet
from data_ver3 import DataModule
import PIL.Image as Image
import os
import numpy as np
import torch

def packaging(model_path = "models/"):
    model_list =  os.listdir(model_path)
    model_name = model_list[-2]
    img_datamodule = DataModule(batch_size=1)
    model = LitResnet.load_from_checkpoint(model_path+os.sep+model_name)

    img_sample = img_datamodule.return_sample()

    img_sample = img_datamodule.transform(img_sample)
    img_sample = img_sample.unsqueeze(0)

    model.to_onnx(
        "models/model.onnx", 
        img_sample,
        export_params=True,
        input_names = ['input'],    # Input names
        output_names = ['output']
    )

if __name__ == "__main__":
    packaging()

