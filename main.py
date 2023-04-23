#import torch
import uvicorn
import numpy as np
import onnxruntime

#from torch import Tensor
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel


class ItemIn(BaseModel):
    accs:List[float]
    gyros:List[float]



app = FastAPI()
#model = torch.jit.load('model.pth')
#model.eval()

model = onnxruntime.InferenceSession('model.onnx')
model_input = model.get_inputs()[0].name


def model_predict(row, model):
    row = np.array(row).reshape(1, -1).astype(np.float32)
    yprev = model.run(None, {model_input: row})
    yprev = np.argmax(yprev)
    
    #row = Tensor([row])
    #yprev = model(row)
    #yprev = np.argmax(yprev.detach().numpy(), axis=1)[0]

    return yprev

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model=List[int])
async def predict(items:List[List[ItemIn]]):
    
    most_frequent = lambda l: max(set(l), key = l.count)

    predictions = []

    for block in items:
        block_predictions = []
        
        for i in block:
            row = i.accs + i.gyros
            pred = model_predict(row, model)
            block_predictions.append(pred)

        predictions.append(most_frequent(block_predictions))
            

    return predictions



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
