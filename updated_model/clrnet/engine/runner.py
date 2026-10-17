import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
try:
    from mmcv.parallel import MMDataParallel
except ModuleNotFoundError:
    from clrnet.utils.mmcv_compat import MMDataParallel


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def _masked_view(self, data):
        masked = dict(data)
        image = data['img'].clone()
        _, _, height, width = image.shape
        lanes = data.get('lane_line')
        if not isinstance(lanes, torch.Tensor):
            masked['img'] = image
            return masked

        for batch_index in range(image.shape[0]):
            valid = torch.nonzero(lanes[batch_index, :, 1] > 0.5,
                                  as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            lane_index = int(valid[torch.randint(valid.numel(), (1,))])
            lane = lanes[batch_index, lane_index]
            length = max(2, min(int(round(float(lane[5]))), self.cfg.num_points))
            fraction = float(np.random.uniform(
                self.cfg.occlusion_min_fraction,
                self.cfg.occlusion_max_fraction))
            occlusion_length = max(2, int(round(length * fraction)))
            start_offset = int(np.random.randint(
                0, max(1, length - occlusion_length + 1)))
            end_offset = start_offset + occlusion_length - 1

            first_x = float(lane[6 + start_offset])
            last_x = float(lane[6 + min(end_offset, self.cfg.num_points - 1)])
            center_x = (first_x + last_x) * 0.5
            midpoint = (start_offset + end_offset) // 2
            neighbor_distances = []
            for other_index in valid.tolist():
                if other_index == lane_index:
                    continue
                other_x = float(lanes[batch_index, other_index,
                                       6 + min(midpoint, self.cfg.num_points - 1)])
                if other_x >= 0:
                    neighbor_distances.append(abs(other_x - center_x))
            nearest = min(neighbor_distances, default=width * 0.08)
            band_width = int(np.clip(nearest * 0.5, 16, width * 0.25))
            center_x = int(np.clip(center_x, 0, width - 1))
            start_x = max(0, min(width - band_width, center_x - band_width // 2))

            # lane_line offsets are ordered from the bottom toward the horizon.
            y_bottom = int(round(height * (1 - start_offset /
                                            max(1, self.cfg.num_points - 1))))
            y_top = int(round(height * (1 - (end_offset + 1) /
                                         max(1, self.cfg.num_points - 1))))
            y_top, y_bottom = sorted((max(0, y_top), min(height, y_bottom)))
            y_bottom = max(y_top + 1, y_bottom)
            image[batch_index, :, y_top:y_bottom, start_x:start_x + band_width] = 0
        masked['img'] = image
        return masked

    def _slice_batch(self, data, start, end):
        batch_size = data['img'].shape[0]
        sliced = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                sliced[key] = value[start:end]
            elif isinstance(value, list) and len(value) == batch_size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _consistency_loss(self, clean, masked):
        total = clean[0].new_zeros(())
        for clean_stage, masked_stage in zip(clean, masked):
            clean_prob = torch.softmax(clean_stage[..., :2].detach(), dim=-1)
            masked_log_prob = torch.log_softmax(masked_stage[..., :2], dim=-1)
            cls = -(clean_prob * masked_log_prob).sum(dim=-1).mean()
            positive = (clean_prob[..., 1] > clean_prob[..., 0])
            reg_values = torch.nn.functional.smooth_l1_loss(
                masked_stage[..., 2:], clean_stage[..., 2:].detach(),
                reduction='none').mean(dim=-1)
            reg = reg_values[positive].mean() if positive.any() else (
                reg_values.mean() * 0)
            total = total + cls + reg
        return total / len(clean)

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            self.optimizer.zero_grad()
            batch_size = data['img'].shape[0]
            microbatch_size = 4 if getattr(
                self.cfg, 'consistency_training', False) else batch_size
            output = None
            accumulated_loss = data['img'].new_zeros(())
            for start in range(0, batch_size, microbatch_size):
                micro = self._slice_batch(
                    data, start, min(start + microbatch_size, batch_size))
                output = self.net(micro)
                loss = output['loss'].sum()
                if getattr(self.cfg, 'consistency_training', False):
                    masked_output = self.net(self._masked_view(micro))
                    consistency = self._consistency_loss(
                        output['predictions_lists'],
                        masked_output['predictions_lists'])
                    ramp = min(1.0, float(epoch + 1) /
                               float(self.cfg.consistency_ramp_epochs))
                    loss = loss + masked_output['loss'].sum() + (
                        self.cfg.consistency_lambda_max * ramp * consistency)
                    output['loss_stats']['masked_loss'] = masked_output['loss']
                    output['loss_stats']['consistency_loss'] = consistency
                    output['loss_stats']['loss'] = loss
                scale = batch_size / microbatch_size
                (loss / scale).backward()
                accumulated_loss = accumulated_loss + loss.detach() / scale
            loss = accumulated_loss
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()


    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        metric = self.test_loader.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)
