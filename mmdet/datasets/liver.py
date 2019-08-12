import numpy as np
from pycocotools.coco import COCO
from .custom import CustomDataset
import os.path as osp
import mmcv
from mmcv.parallel import DataContainer as DC
from .utils import to_tensor, random_scale
from .med_img_aug import MedImgAugmentation

import matplotlib.pyplot as plt

class LiverDataset(CustomDataset):

    CLASSES = ('lesion', 'liver', 'body')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 extra_aug=None,
                 extra_med_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        super().__init__(ann_file=ann_file,
                         img_prefix=img_prefix,
                         img_scale=img_scale,
                         img_norm_cfg=img_norm_cfg,
                         size_divisor=size_divisor,
                         proposal_file=proposal_file,
                         num_max_proposals=num_max_proposals,
                         flip_ratio=flip_ratio,
                         with_mask=with_mask,
                         with_crowd=with_crowd,
                         with_label=with_label,
                         extra_aug=extra_aug,
                         resize_keep_ratio=resize_keep_ratio,
                         test_mode=test_mode)
        self.extra_med_aug = None
        if extra_med_aug is not None:
            self.extra_med_aug = MedImgAugmentation(**extra_med_aug)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def prepare_train_img(self, idx, show=False):
        img_info = self.img_infos[idx]
        # load image
        img_full_path = osp.join(self.img_prefix, img_info['filename'])
        _, file_extension = osp.splitext(img_full_path)
        if file_extension == '.npy':
            img = np.load(img_full_path)
        else:
            img = mmcv.imread(img_full_path)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        gt_masks = ann['masks']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        img, gt_masks = self.pad_img_mask(img, gt_masks)

        # extra augmentation
        if self.extra_aug is not None:                   # Not Used
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)
        # extra augmentation for medical images
        if self.extra_med_aug is not None:
            img, gt_masks, gt_bboxes = self.extra_med_aug(img, gt_masks, gt_bboxes)

        if show:
            plt.figure(1)
            plt.subplot(121)
            plt.imshow(img[:, :, :]/255)
            plt.subplot(122)
            plt.imshow(gt_masks[0, :, :])
            plt.show()

        # apply transforms
        flip = False
        img_scale = self.img_scales[0]            #random_scale(self.img_scales)  # sample a scale
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)         # img_shape ((1024, 1024, 3)) pad_shape ((1024, 1024, 3)  scale_factor(1.0)

        img = img.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(gt_masks, pad_shape, scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        img_plot = np.transpose(img, (1, 2, 0))
        if show:
            plt.figure(2)
            plt.subplot(121)
            plt.imshow(img_plot[:, :, :])
            plt.subplot(122)
            plt.imshow(gt_masks[0, :, :])
            plt.show()

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img_full_path = osp.join(self.img_prefix, img_info['filename'])
        _, file_extension = osp.splitext(img_full_path)
        if file_extension == '.npy':
            img = np.load(img_full_path)
        else:
            img = mmcv.imread(img_full_path)
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

    def pad_img_mask(self, img, mask):
        (w, h) = (img.shape[0], img.shape[1])
        if w == h:
            return img, mask
        elif w > h:
            new_img = np.zeros(shape=(w,w,3), dtype=np.uint8)
            new_img[0: w, 0: h, :] = img
            new_mask = []
            for i in range(len(mask)):
                curr_mask = np.zeros(shape=(w,w), dtype=np.uint8)
                curr_mask[0: w, 0: h] = mask[i]
                new_mask.append(curr_mask)
        elif w < h:
            new_img = np.zeros(shape=(h,h,3), dtype=np.uint8)
            new_img[0: w, 0: h, :] = img
            new_mask = []
            for i in range(len(mask)):
                curr_mask = np.zeros(shape=(h,h), dtype=np.uint8)
                curr_mask[0: w, 0: h] = mask[i]
                new_mask.append(curr_mask)

        return new_img, new_mask
