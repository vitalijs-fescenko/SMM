import sys
#from segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.model.build_sam_encoder import sam_encoder_model_registry
from src.model.smm import SMM

sam_enc_checkpoint = "weights/sam_vit_b_01ec64_encoder.pth"
model_type = "vit_b_encoder"

# load SAM encoder
device = "cpu"
sam_encoder = sam_encoder_model_registry[model_type](checkpoint=sam_enc_checkpoint)

# create SMM model
smm = SMM(sam_encoder)
smm.to(device=device)

image = cv2.imread('images/flowers.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
embeddings = smm._forward_encoder(image)

print(embeddings.shape)