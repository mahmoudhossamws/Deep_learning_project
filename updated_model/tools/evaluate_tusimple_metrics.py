import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clrnet.utils.tusimple_metric import LaneEval


PIXEL_THRESHOLD = 20
POINT_ACCURACY_THRESHOLD = 0.85


def lane_points(lane, y_samples, mask=None):
    points = [(x, y) for x, y in zip(lane, y_samples) if x >= 0]
    if mask is not None:
        points = [(x, y) for x, y in points if mask(x, y)]
    return points


def line_accuracy(pred, gt, y_samples, mask=None):
    pairs = [(p, g) for p, g, y in zip(pred, gt, y_samples)
             if (p >= 0 or g >= 0) and (mask is None or mask(
                 p if p >= 0 else g, y))]
    if not pairs:
        return None
    correct = sum(p >= 0 and g >= 0 and abs(p - g) < PIXEL_THRESHOLD
                  for p, g in pairs)
    return correct / len(pairs)


def evaluate_frame(prediction, ground_truth, mask):
    y_samples = ground_truth['h_samples']
    gt_lanes = ground_truth['lanes']
    pred_lanes = prediction['lanes']
    accuracy, false_positive, false_negative = LaneEval.bench(
        pred_lanes, gt_lanes, y_samples, prediction['run_time'])

    masked_gt = [
        lane for lane in gt_lanes
        if lane_points(lane, y_samples, mask)
    ]
    masked_accuracies = []
    for gt_lane in masked_gt:
        values = [
            value for value in
            (line_accuracy(pred_lane, gt_lane, y_samples, mask)
             for pred_lane in pred_lanes) if value is not None
        ]
        masked_accuracies.append(max(values, default=0.0))

    masked_detected = sum(
        value >= POINT_ACCURACY_THRESHOLD for value in masked_accuracies)
    masked_points = [
        max(values, default=0.0) for gt_lane in masked_gt
        for values in [[
            value for value in
            (line_accuracy(pred_lane, gt_lane, y_samples, mask)
             for pred_lane in pred_lanes) if value is not None
        ]]
    ]
    return (accuracy, false_positive, false_negative, masked_detected,
            len(masked_gt), masked_points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.predictions) as source:
        predictions = [json.loads(line) for line in source if line.strip()]
    with open(args.labels) as source:
        labels = [json.loads(line) for line in source if line.strip()]
    labels_by_file = {item['raw_file']: item for item in labels}
    mask = lambda x, y: 480 <= x <= 800 and 300 <= y <= 500

    totals = [0.0] * 5
    masked_points = []
    for prediction in predictions:
        result = evaluate_frame(prediction, labels_by_file[prediction['raw_file']],
                                mask)
        for index, value in enumerate(result[:5]):
            totals[index] += value
        masked_points.extend(result[5])

    count = len(predictions)
    accuracy, fp, fn, masked_detected, masked_total = [
        value / count for value in totals
    ]
    precision = 1 - fp
    recall = 1 - fn
    output = {
        'accuracy': accuracy,
        'false_positives': fp,
        'false_negatives': fn,
        'f1_score': 2 * precision * recall / max(precision + recall, 1e-12),
        'masked_lane_recall': masked_detected / max(masked_total, 1e-12),
        'masked_segment_accuracy': float(np.mean(masked_points))
        if masked_points else 0.0,
        'num_frames': count,
        'protocol': {
            'pixel_threshold': PIXEL_THRESHOLD,
            'lane_point_accuracy_threshold': POINT_ACCURACY_THRESHOLD,
            'synthetic_mask': {'x_min': 480, 'x_max': 800,
                               'y_min': 300, 'y_max': 500},
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as destination:
        json.dump(output, destination, indent=2)
        destination.write('\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
