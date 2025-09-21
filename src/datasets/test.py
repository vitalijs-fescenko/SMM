from panopticapi.utils import rgb2id
import cv2
from src.target_generator import TargetGenerator

class TestDataset:
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path
        self.target_generator = TargetGenerator()

    def load_sample(self):
        image = cv2.imread(self.img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label_id = rgb2id(label)

        return image, label_id

    def convert_2_target(self, label):
        return self.target_generator(label)


