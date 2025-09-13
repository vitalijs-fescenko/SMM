from segment_anything import sam_model_registry, SamPredictor
import torch

# load SAM
sam_checkpoint = "weights/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)


# save only-encoder
save_path = "weights/sam_vit_b_01ec64_encoder.pth"
torch.save(sam.image_encoder.state_dict(), save_path)

