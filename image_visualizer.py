import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SegmentationVisualizer:
    def __init__(self):
        self.categories = {
            1: {"name": "dry", "color": [244, 164, 96]},
            2: {"name": "humid", "color": [135, 206, 235]},
            3: {"name": "slush", "color": [112, 128, 144]},
            4: {"name": "snow", "color": [255, 255, 255]},
            5: {"name": "wet", "color": [70, 130, 180]}
        }

    def load_json_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return None

    def process_prediction_data(self, json_data):
        try:
            if 'load_segmentation' in json_data and 'prediction' in json_data['load_segmentation']:
                prediction_data = json_data['load_segmentation']['prediction']
                return np.array(prediction_data)
            else:
                logger.warning("Invalid JSON structure: Missing required keys")
                return None
        except Exception as e:
            logger.error(f"Error processing prediction data: {e}")
            return None

    def create_colored_mask(self, segmentation_data):
        height, width = segmentation_data.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, info in self.categories.items():
            mask = segmentation_data == class_id
            colored_mask[mask] = info['color']

        return colored_mask

    def visualize(self, image_path, segmentation_data, alpha=0.5, save_path=None):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return

        seg_resized = cv2.resize(segmentation_data, (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        colored_mask = self.create_colored_mask(seg_resized)
        overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(colored_mask)
        plt.title('Segmentation Mask')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')

        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=np.array(info['color']) / 255)
                           for info in self.categories.values()]
        legend_labels = [info['name'] for info in self.categories.values()]
        plt.figlegend(legend_elements, legend_labels, loc='center right',
                      bbox_to_anchor=(1.2, 0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def print_statistics(self, segmentation_data):
        unique, counts = np.unique(segmentation_data, return_counts=True)
        total_pixels = segmentation_data.size

        print("\nSegmentation Statistics:")
        print(f"Image Shape: {segmentation_data.shape}")
        print("\nClass Distribution:")

        for value, count in zip(unique, counts):
            value = int(value)
            percentage = (count / total_pixels) * 100
            name = "background" if value == 0 else self.categories[value]['name']
            print(f"{name}: {count} pixels ({percentage:.2f}%)")


def process_single_json(json_path, visualizer, output_dir):
    try:
        json_data = visualizer.load_json_file(json_path)
        if not json_data:
            return

        image_path = json_data.get('image_path')
        if not image_path:
            logger.warning(f"No image path in JSON: {json_path}")
            return

        segmentation_data = visualizer.process_prediction_data(json_data)
        if not segmentation_data is not None:
            return

        vis_output_path = output_dir / f'{json_path.stem}.png'

        visualizer.print_statistics(segmentation_data)
        visualizer.visualize(image_path, segmentation_data, alpha=0.5, save_path=str(vis_output_path))
        logger.info(f"Saved to: {vis_output_path}")

    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")


def main():
    base_dir = Path("/home/allbigdat/data")
    results_dir = base_dir / "inference_results_test"
    vis_output_dir = base_dir / "visualizations"
    vis_output_dir.mkdir(exist_ok=True)

    visualizer = SegmentationVisualizer()

    json_files = list(results_dir.glob('*.json'))
    for json_path in tqdm(json_files):
        process_single_json(json_path, visualizer, vis_output_dir)


if __name__ == "__main__":
    main()