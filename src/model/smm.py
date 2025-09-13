import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple
import numpy as np


from segment_anything.modeling.image_encoder import ImageEncoderViT

from segment_anything.utils.transforms import ResizeLongestSide

# adapted from segment_anything.modeling.sam

class SMM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SMM predicts object masks from an image.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.transform = ResizeLongestSide(self.image_encoder.img_size)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        # for image_record, curr_embedding in zip(batched_input, image_embeddings):
        #     if "point_coords" in image_record:
        #         points = (image_record["point_coords"], image_record["point_labels"])
        #     else:
        #         points = None
        #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #         points=points,
        #         boxes=image_record.get("boxes", None),
        #         masks=image_record.get("mask_inputs", None),
        #     )
        #     low_res_masks, iou_predictions = self.mask_decoder(
        #         image_embeddings=curr_embedding.unsqueeze(0),
        #         image_pe=self.prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=multimask_output,
        #     )
        #     masks = self.postprocess_masks(
        #         low_res_masks,
        #         input_size=image_record["image"].shape[-2:],
        #         original_size=image_record["original_size"],
        #     )
        #     masks = masks > self.mask_threshold
        #     outputs.append(
        #         {
        #             "masks": masks,
        #             "iou_predictions": iou_predictions,
        #             "low_res_logits": low_res_masks,
        #         }
        #     )
        return outputs


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


    def _forward_encoder(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        assert (
            len(input_image_torch.shape) == 4
            and input_image_torch.shape[1] == 3
            and max(*input_image_torch.shape[2:]) == self.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.image_encoder.img_size}."

        #original_size = image.shape[:2]
        #input_size = tuple(transformed_image.shape[-2:])
        input_image = self.preprocess(input_image_torch)
        features = self.image_encoder(input_image)
        return features

