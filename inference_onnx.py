import numpy as np
import onnxruntime as ort
from scipy.special import softmax
import torch
from model_ver2 import LitResnet
from data_ver3 import DataModule

class ONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.lables = ["0", "1"]

    def predict(self, img):
        processed = self.processor.transform(img)
        processed = processed.unsqueeze(0)
        logits = self.ort_session.run(None, {'input' : processed.numpy()})
        scores = softmax(logits[0])[0]
        if scores[0] > scores[1]:
            output = "cat"
        else:
            output = "dog"
        score= np.max(scores)
        return {"result": output, "score":score}
    
if __name__ == "__main__":
    predictor = ONNXPredictor("models/model.onnx")
    img_datamodule = DataModule()
    img_sample = img_datamodule.return_sample()
    
    result = predictor.predict(img_sample)
    print(result)