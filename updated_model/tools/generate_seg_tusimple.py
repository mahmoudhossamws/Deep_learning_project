import argparse
import json
import os

import cv2
import numpy as np


ORIGINAL_SPLITS = {
    'train_val': (['label_data_0313.json', 'label_data_0601.json',
                   'label_data_0531.json'], ''),
    'test': (['test_label.json'], ''),
}
LOCAL_SPLITS = {
    'train': (['train/labels.json'], 'train'),
    'val': (['val/labels.json'], 'val'),
    'test': (['test/labels.json'], 'test'),
}


def _ordered_lanes(label):
    lanes = []
    candidates = []
    slopes = []
    for lane, y_samples in zip(label['lanes'], [label['h_samples']] *
                               len(label['lanes'])):
        points = [(x, y) for x, y in zip(lane, y_samples) if x >= 0]
        if len(points) > 1:
            candidates.append(points)
            slopes.append(np.arctan2(points[-1][1] - points[0][1],
                                     points[0][0] - points[-1][0]) /
                          np.pi * 180)

    order = np.argsort(slopes)
    candidates = [candidates[i] for i in order]
    slopes = [slopes[i] for i in order]
    indices = [None] * 6
    for i, slope in enumerate(slopes):
        if slope <= 90:
            indices[2] = i
            indices[1] = i - 1 if i > 0 else None
            indices[0] = i - 2 if i > 1 else None
        else:
            indices[3] = i
            indices[4] = i + 1 if i + 1 < len(slopes) else None
            indices[5] = i + 2 if i + 2 < len(slopes) else None
            break

    return [([] if index is None else candidates[index])
            for index in indices]


def generate_split(root, savedir, annotation_files, image_root):
    for annotation_file in annotation_files:
        with open(os.path.join(root, annotation_file)) as source:
            for line in source:
                label = json.loads(line)
                segmentation = np.zeros((720, 1280, 3), dtype=np.uint8)
                for lane_index, lane in enumerate(_ordered_lanes(label)):
                    if len(lane) < 4:
                        continue
                    for start, end in zip(lane, lane[1:]):
                        cv2.line(segmentation, start, end,
                                 (lane_index + 1,) * 3, 15)

                relative_path = label['raw_file'].replace(
                    'clips', savedir, 1)
                output_path = os.path.join(root, image_root, relative_path)
                output_path = os.path.splitext(output_path)[0] + '.png'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, segmentation)


def main():
    parser = argparse.ArgumentParser(
        description='Generate CLRNet TuSimple segmentation labels.')
    parser.add_argument('--root', required=True,
                        help='Root of the TuSimple dataset.')
    parser.add_argument('--savedir', default='seg_label',
                        help='Segmentation directory name.')
    args = parser.parse_args()

    splits = LOCAL_SPLITS if os.path.isfile(
        os.path.join(args.root, 'train', 'labels.json')) else ORIGINAL_SPLITS
    for split, (annotation_files, image_root) in splits.items():
        print('generating {} set...'.format(split))
        generate_split(args.root, args.savedir, annotation_files, image_root)


if __name__ == '__main__':
    main()
