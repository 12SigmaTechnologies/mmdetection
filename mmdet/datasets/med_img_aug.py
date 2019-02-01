import numpy as np
from externals.batchgenerators.batchgenerators.transforms.spatial_transforms import SpatialTransform
from externals.batchgenerators.batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from externals.batchgenerators.batchgenerators.transforms.crop_and_pad_transforms import CenterCropSegTransform
import matplotlib.pyplot as plt

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rows_exist = np.where(rows)
    cols_exist = np.where(cols)
    if len(rows_exist[0]) == 0 or len(cols_exist[0]) == 0:
        return 0, 0, 0, 0
    rmin, rmax = rows_exist[0][[0, -1]]
    cmin, cmax = cols_exist[0][[0, -1]]
    return cmin, rmin, cmax+1, rmax+1

class MedImgAugmentation(object):

    def __init__(self, spatial_config=None, crop_config=None, noise_config=None):
        self.transforms = []
        if spatial_config is not None:
            self.transforms.append(SpatialTransform(**spatial_config))
        if crop_config is not None:
            self.transforms.append(CenterCropSegTransform(**crop_config))
        if noise_config is not None:
            self.transforms.append(GaussianNoiseTransform(**noise_config))

    def __call__(self, img, mask, boxes):
        img = img.astype(np.float32)
        mask = np.array(mask)
        show = False
        if show:
            plt.figure(1)
            plt.subplot(221)
            plt.imshow(img[:, :, 2:5]/255)
            plt.subplot(222)
            plt.imshow(mask[0, :, :])
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
        mask = np.expand_dims(mask, 0)
        data_dict = {"data": img, "seg": mask}
        for transform in self.transforms:
            data_dict = transform(**data_dict)
        new_img = np.transpose(np.squeeze(data_dict["data"], 0), (1, 2, 0))
        new_mask = np.squeeze(data_dict["seg"], 0)
        mask_num = new_mask.shape[0]
        new_boxes_list = []
        for i in range(mask_num):
            cur_mask = new_mask[i]
            cur_new_box = bbox2(cur_mask)
            new_boxes_list.append(cur_new_box)
        new_boxes = np.array(new_boxes_list, dtype=np.float32)
        if show:
            plt.subplot(223)
            plt.imshow(new_img[:, :, 2:5]/255)
            plt.subplot(224)
            plt.imshow(new_mask[0, :, :])
            plt.show()
        # TODO: change the boxes_ignore accordingly
        return new_img, new_mask, new_boxes
