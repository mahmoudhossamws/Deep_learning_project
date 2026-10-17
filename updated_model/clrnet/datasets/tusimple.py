import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
from clrnet.utils.tusimple_metric import LaneEval
from .registry import DATASETS
import logging
import random

SPLIT_FILES = {
    'trainval':
    ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

LOCAL_SPLIT_FILES = {
    'train': ('train', 'labels.json'),
    'val': ('val', 'labels.json'),
    'test': ('test', 'labels.json'),
}


@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes, cfg)
        self.split = split
        self.anno_files = SPLIT_FILES[split]
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.data_infos = []
        max_lanes = 0
        annotation_files = self._annotation_files()
        for anno_file, split_root in annotation_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                raw_file = data['raw_file']
                mask_path = raw_file.replace('clips', 'seg_label')[:-3] + 'png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                         for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path':
                    osp.join(split_root, raw_file),
                    'img_name':
                    osp.relpath(osp.join(split_root, raw_file), self.data_root),
                    'raw_file':
                    raw_file,
                    'mask_path':
                    osp.join(split_root, mask_path),
                    'lanes':
                    lanes,
                })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def _annotation_files(self):
        local_split = LOCAL_SPLIT_FILES.get(self.split)
        if local_split is not None:
            split, filename = local_split
            local_file = osp.join(self.data_root, split, filename)
            if osp.isfile(local_file):
                return [(local_file, osp.join(self.data_root, split))]
            raise FileNotFoundError(
                f'Expected local TuSimple annotations at {local_file}. '
                f'Check TUSIMPLE_ROOT and the dataset layout.')

        return [(osp.join(self.data_root, filename), self.data_root)
                for filename in self.anno_files]

    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.cfg.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['raw_file']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions,
                                                        runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir,
                                     'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        split_gt_file = osp.join(self.data_root, self.split, 'labels.json')
        gt_file = (split_gt_file
                   if osp.isfile(split_gt_file)
                   else self.cfg.test_json_file)
        result, acc = LaneEval.bench_one_submit(pred_filename, gt_file)
        self.logger.info(result)
        return acc
