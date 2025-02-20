import os
from pathlib import Path
import json
import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor
from tqdm import tqdm

from deeplabv3.models import get_model
import mmcv
from mmdet.apis import init_detector, inference_detector
from typing import Dict, Any, Optional, List
import logging

from resource.loads.model_cfg import MODEL_CFG, MODEL_TYPE, \
    __PERSONAL_LABEL_MAP__, __LOAD_LABEL_MAP__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchInferenceProcessor:
    def __init__(self, test_json_path: str):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._init_models()
        self.test_data = self._load_test_data(test_json_path)

    def _init_models(self):
        # Personal detection model initialization
        self.personal_model = init_detector(
            "resource/personal/moveawheel_faster_rcnn_cfg.py",
            "resource/personal/personal_weight.pth",
            device=self.device
        )
        self.personal_model.eval()

        # Load segmentation model initialization
        self.load_model = get_model(MODEL_TYPE, MODEL_CFG)
        load_model_pth = torch.load(
            "resource/loads/model_weights.pth",
            map_location=torch.device(self.device)
        )
        self.load_model.load_state_dict(load_model_pth)
        self.load_model.eval()
        self.load_model.to(self.device)

    def _load_test_data(self, json_path: str) -> List[Dict]:
        """Load and parse the test data JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return [{'file_name': img['file_name'], 'sensor_info': img['sensor_info']}
                for img in data['images']]

    def _personal_postprocess(self, result):
        return {
            "boxes": result[0].pred_instances.bboxes.detach().cpu().numpy().tolist(),
            "labels": [__PERSONAL_LABEL_MAP__[idx]
                       for idx in result[0].pred_instances.labels.detach().cpu().numpy().tolist()],
            "scores": result[0].pred_instances.scores.detach().cpu().numpy().tolist(),
        }

    def _loads_postprocess(self, result):
        return {
            "prediction": torch.argmax(result, dim=0).cpu().numpy().tolist()
        }

    def process_load_image(self, image_path: str, sensor_data: Dict) -> Dict[str, Any]:
        # Read and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = mmcv.imconvert(image, 'bgr', 'rgb')

        # Load segmentation
        with torch.no_grad():
            image_tensor = ToTensor()(cv2.resize(image, (520, 520)))
            sensors = torch.Tensor([
                sensor_data["humi"],
                sensor_data["pressure"],
                sensor_data["objectTemp"],
                sensor_data["latitude"],
                sensor_data["longitude"],
                sensor_data["height"],
            ])

            image_tensor = image_tensor.to(self.device)
            sensors = sensors.to(self.device)

            load_pred = self.load_model(
                self.load_model,
                image_tensor.unsqueeze(0),
                sensors.unsqueeze(0)
            )['out'][0]
            load_result = self._loads_postprocess(load_pred)

        return {
            "image_path": image_path,
            "load_segmentation": load_result
        }

    def process_all(self, base_image_dir: str, output_dir: str):
        """Process all test images"""
        for test_item in tqdm(self.test_data):
            image_path = os.path.join(base_image_dir, test_item['file_name'])
            output_path = os.path.join(output_dir, f"{os.path.splitext(test_item['file_name'])[0]}.json")

            try:
                result = self.process_load_image(image_path, test_item['sensor_info'])

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f"Processed {test_item['file_name']}")

            except Exception as e:
                logger.error(f"Error processing {test_item['file_name']}: {str(e)}")
                continue


def main():
    base_dir = "/home/allbigdat/data/"
    test_json_path = os.path.join(base_dir, "COCO", "test_without_street.json")
    image_dir = os.path.join(base_dir, "images")
    results_dir = os.path.join(base_dir, "inference_results")

    processor = BatchInferenceProcessor(test_json_path)
    processor.process_all(image_dir, results_dir)


if __name__ == "__main__":
    main()