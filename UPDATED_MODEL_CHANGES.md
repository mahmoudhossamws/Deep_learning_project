# Baseline to Updated CLRNet Changes

This document describes the code changes made in `updated_model/` relative to
the working CLRNet baseline. The goal was to implement the three proposed
research updates while preserving the baseline behavior when the corresponding
configuration switches are disabled.

## 1. Visibility-aware ROI gating

### Baseline

The baseline `ROIGather` module processed each pooled lane ROI with convolution
layers, flattened the result, and projected it through a fully connected layer.
Every sampled ROI feature contributed equally to that projection.

### Updated implementation

In [roi_gather.py](updated_model/clrnet/models/utils/roi_gather.py), the
updated model adds one visibility gate per refinement stage:

```python
gate = torch.sigmoid(self.visibility_gates[layer_index](roi))
roi = roi * gate
```

Each gate is produced by a small `1x1` convolutional bottleneck and has one
output channel. Therefore, the gate is a scalar spatial weight for each lane
ROI sample, rather than a separate unrestricted weight for every feature
channel. The final gate bias is initialized so that the initial sigmoid value
is approximately `0.95`; this lets training begin close to the baseline
feature scale instead of suppressing all ROI features at initialization.

The feature is controlled by:

```python
use_visibility_gate = True
```

The default constructor value is `False`, so the original baseline path is
available for ablation.

## 2. GMP-ECA channel attention

### Baseline

The baseline passed the gathered ROI tensor directly to the fully connected
projection after ROI convolution processing.

### Updated implementation

The updated `ROIGather` adds efficient channel attention after visibility
gating. It computes a channel descriptor using global max pooling by default,
then applies a lightweight one-dimensional convolution and sigmoid:

```python
descriptor = roi.amax(dim=(2, 3)).unsqueeze(1)
scale = torch.sigmoid(self.gmp_eca[layer_index](descriptor)).squeeze(1)
roi = roi * scale.unsqueeze(-1).unsqueeze(-1) + roi
```

The residual addition preserves the original ROI representation while allowing
the attention module to amplify useful channels. Average pooling is also
available for ablation through `use_avg_eca`, but the completed run used GMP:

```python
use_gmp_eca = True
use_avg_eca = False
```

The `CLRHead` passes these configuration flags into `ROIGather` in
[clr_head.py](updated_model/clrnet/models/heads/clr_head.py).

## 3. Occlusion-consistency training

### Baseline

The baseline used one augmented image view and optimized its supervised CLRNet
loss:

```python
output = self.net(data)
loss = output['loss'].sum()
loss.backward()
```

### Updated implementation

The updated runner in [runner.py](updated_model/clrnet/engine/runner.py)
creates a second view of each training sample. The second view masks a
contiguous part of an annotated lane:

1. Select a valid annotated lane.
2. Select a contiguous lane segment covering a random 10--50% of its valid
   vertical extent.
3. Estimate the mask width from the nearest neighboring lane, with minimum and
   maximum bounds.
4. Paint that image band with the neutral value used by the normalized input.

The clean and masked views both receive the normal supervised loss. The runner
also adds a consistency loss between their predictions. Clean predictions are
detached and used as the teacher target:

- Classification consistency is computed over all lane priors.
- Regression consistency is computed only for clean-view positive priors.
- The consistency coefficient is linearly ramped from zero to its configured
  maximum over the first ten epochs.

The relevant configuration is:

```python
consistency_training = True
consistency_lambda_max = 1.0
consistency_ramp_epochs = 10
occlusion_min_fraction = 0.10
occlusion_max_fraction = 0.50
```

## 4. Supporting changes required to run the update

The two-view training pass temporarily needs more memory than the baseline.
The updated runner therefore splits each batch into microbatches of four,
performs clean and masked forward passes for each microbatch, scales each loss
before backpropagation, and retains an effective batch size of eight. This
avoids the CUDA out-of-memory failure encountered when both complete graphs
were held at once.

The updated `CLRHead` also returns `predictions_lists` along with the normal
loss and loss statistics. This exposes the intermediate refinement-stage
predictions required by the consistency loss without changing inference
behavior.

## 5. Configuration and reproducibility

The switches are defined in
[clr_resnet34_tusimple.py](updated_model/configs/clrnet/clr_resnet34_tusimple.py).
Setting all three update mechanisms to `False` restores the baseline
architecture/training path, apart from the compatibility and dataset-loader
changes shared by both projects.

## 6. Completed run

The corrected 70-epoch run was saved under:

`updated_model/work_dirs/clr/r34_tusimple_updated_fixed/20261016_223448_lr_8e-04_b_8/`

Results from the final test evaluation:

- Accuracy: `0.9580188114`
- False positives: `0.0486400671`
- False negatives: `0.0291157441`
- F1 score: `0.9610229397`
- Masked-lane recall: `0.9526980874`
- Masked-segment accuracy: `0.9654189196`

The final checkpoint is `ckpt/69.pth`. The run log, metrics, predictions, and
training summary are stored in the same run directory.
