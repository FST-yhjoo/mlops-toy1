from fastapi import FastAPI
# from inference_onnx import ColaONNXPredictor
from inference import Predictor
app = FastAPI(title="MLOps Basics App")

predictor = Predictor("models/model.onnx")

@app.get("/")
async def home_page():
    return 'Sample prediction API'


@app.get("/predict")
async def get_prediction(text: str):
    result =  predictor.predict(text)
    return result