from typing import Dict, Any, Annotated
from io import BytesIO
import cv2
import torch
import uvicorn
import numpy as np
from torchvision.transforms import ToTensor
from deeplabv3.models import get_model

import torchvision.transforms as transforms
from fastapi import FastAPI
from fastapi import UploadFile
from PIL import Image

from resource.loads.model_cfg import MODEL_CFG, MODEL_TYPE,\
    __PERSONAL_LABEL_MAP__, __LOAD_LABEL_MAP__
import mmcv
from mmdet.apis import init_detector, inference_detector


app = FastAPI()
transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
])

personal_device = "cuda:0"
load_device = "cuda:0"

personal_model = init_detector("resource/personal/moveawheel_faster_rcnn_cfg.py", 
                               "resource/personal/personal_weight.pth",
                               device=personal_device)
print("test3")
personal_model = personal_model.eval()

load_model = get_model(MODEL_TYPE, MODEL_CFG)
load_model_pth = torch.load("resource/loads/model_weights.pth",
                            map_location=torch.device(load_device))
load_model.load_state_dict(load_model_pth)
load_model = load_model.eval()
load_model.to(personal_device)


def _personal_postprocess(result):
    return {
            "boxes": result[0].pred_instances.bboxes.detach().cpu().numpy().tolist(),
            "labels": [__PERSONAL_LABEL_MAP__[idx] \
                for idx in result[0].pred_instances.labels.detach().cpu().numpy().tolist()],
            "scores": result[0].pred_instances.scores.detach().cpu().numpy().tolist(),
    }


def _predict_personal(image):
    with torch.no_grad():
        output = inference_detector(personal_model, [image])
        return output


@app.post("/pred_personal")
async def predict_personal(file: UploadFile):
    contents = await file.read()
    
    image = Image.open(BytesIO(contents))
    image = np.array(image)
    image = mmcv.imconvert(image, 'bgr', 'rgb')
    prediction = _predict_personal(image)
    result = _personal_postprocess(prediction)

    return {
        "filename": file.filename, 
        "prediction": result,
    }


def _loads_postprocess(result):
    
    return {
        "prediction": torch.argmax(result, dim=0).cpu().numpy().tolist()
    }


def _predict_loads(image, sensors):
    with torch.no_grad():
        image = ToTensor()(cv2.resize(image, (520, 520)))
        sensors = torch.Tensor(
            [
                sensors["humi"],
                sensors["pressure"],
                sensors["objectTemp"],
                sensors["latitude"],
                sensors["longitude"],
                sensors["height"],
            ]
        )
        image = image.to(load_device)
        sensors = sensors.to(load_device)

        output = load_model(load_model, image.unsqueeze(0), sensors.unsqueeze(0))['out'][0]
        return output


@app.post("/pred_loads")
async def predict_loads(file: UploadFile,
                        objectTemp: float,
                        humi: float,
                        pressure:float,
                        latitude:float,
                        longitude:float,
                        height:float,
                        ):
    contents = await file.read()
    
    image = Image.open(BytesIO(contents))
    image = np.array(image)
    image = mmcv.imconvert(image, 'bgr', 'rgb')
    sensors = dict(
        objectTemp= objectTemp,
        humi= humi,
        pressure= pressure,
        latitude= latitude,
        longitude= longitude,
        height= height,
    )
    prediction = _predict_loads(image, sensors)
    result = _loads_postprocess(prediction)

    return {
        "filename": file.filename, 
        "prediction": result,
    }

@app.get("/")
def read_root():
    return {"message": "ABD model API."}

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=48121, reload=True)
    
if __name__ == "__main__":
    main()