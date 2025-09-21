# Adapted from detectron2.projects.Panoptic-DeepLab.panoptic_deeplab.target_generator.py
import numpy as np
import torch


class TargetGenerator:
    """
    Generates training targets for center and offset regression.
    """

    def __init__(
        self,
        sigma=8,
        small_instance_area=0,
        small_instance_weight=1,
    ):
        """
        Args:
            thing_ids: Set, a set of ids from contiguous category ids belonging
                to thing categories.
            sigma: the sigma for Gaussian kernel.
            small_instance_area: Integer, indicates largest area for small instances.
            small_instance_weight: Integer, indicates semantic loss weights for
                small instances.
        """
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight

        # Generate the default Gaussian image for each center
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def __call__(self, instance_map):
        """Generates the training target from instance map.
        Args:
            instance_map: numpy.array, instance_map, we assume it is already
                converted from rgb image by panopticapi.utils.rgb2id.

        Returns:
            A dictionary with fields:
                - sem_seg: Tensor, semantic label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(H, W).
                - center_points: List, center coordinates, with tuple
                    (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is
                    (offset_y, offset_x).
                - sem_seg_weights: Tensor, loss weight for semantic prediction,
                    shape=(H, W).
                - center_weights: Tensor, ignore region of center prediction,
                    shape=(H, W), used as weights for center regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction,
                    shape=(H, W), used as weights for offset regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
        """
        height, width = instance_map.shape[0], instance_map.shape[1]
        center = np.zeros((height, width), dtype=np.float32)

        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
        )
        center_weights = np.zeros_like(instance_map, dtype=np.uint8)
        offset_weights = np.zeros_like(instance_map, dtype=np.uint8)

        for cat_id in np.unique(instance_map):
            if cat_id == 0:
                continue

            # for each instance 
            # find instance center
            ins_mask = instance_map == cat_id
            mask_index = np.where(ins_mask)

            # Find instance area
            #ins_area = len(mask_index[0])

            center_weights[ins_mask] = 1
            offset_weights[ins_mask] = 1

            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            center_pts.append([center_y, center_x])

            # generate center heatmap
            y, x = int(round(center_y)), int(round(center_x))
            sigma = self.sigma
            # upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            # bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            # start and end indices in default Gaussian image
            gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], width) - ul[0]
            gaussian_y0, gaussian_y1 = max(0, -ul[1]), min(br[1], height) - ul[1]

            # start and end indices in center heatmap image
            center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
            center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
            center[center_y0:center_y1, center_x0:center_x1] = np.maximum(
                center[center_y0:center_y1, center_x0:center_x1],
                self.g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
            )

            # generate offset (2, h, w) -> (y-dir, x-dir)
            offset[0][mask_index] = center_y - y_coord[mask_index]
            offset[1][mask_index] = center_x - x_coord[mask_index]

        center_weights = center_weights[None]
        offset_weights = offset_weights[None]
        center = center[None]

        return dict(
            center=torch.as_tensor(center.astype(np.float32)),
            center_points=center_pts,
            offset=torch.as_tensor(offset.astype(np.float32)),
            center_weights=torch.as_tensor(center_weights.astype(np.float32)),
            offset_weights=torch.as_tensor(offset_weights.astype(np.float32)),
        )
