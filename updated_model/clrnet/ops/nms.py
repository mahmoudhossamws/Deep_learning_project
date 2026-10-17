# Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch


try:
    from . import nms_impl
except ImportError:
    nms_impl = None


def nms(boxes, scores, overlap, top_k):
    if nms_impl is not None:
        return nms_impl.nms_forward(boxes, scores, overlap, top_k)

    if boxes.numel() == 0:
        return boxes.new_empty((0,), dtype=boxes.dtype)

    order = scores.argsort(descending=True)
    keep = []
    for current in order:
        if len(keep) >= top_k:
            break
        suppressed = False
        for previous in keep:
            start = max(int(round(float(boxes[current, 2] * 71))),
                        int(round(float(boxes[previous, 2] * 71))))
            end = min(
                71,
                int(round(float(boxes[current, 2] * 71 +
                             boxes[current, 4] - 1))),
                int(round(float(boxes[previous, 2] * 71 +
                             boxes[previous, 4] - 1))))
            if end < start:
                continue
            distance = torch.mean(torch.abs(
                boxes[current, 5 + start:6 + end] -
                boxes[previous, 5 + start:6 + end]))
            if distance < overlap:
                suppressed = True
                break
        if not suppressed:
            keep.append(current)

    keep_tensor = torch.stack(keep) if keep else boxes.new_empty(
        (0,), dtype=torch.long)
    num_to_keep = boxes.new_tensor(len(keep), dtype=torch.long)
    parent_index = boxes.new_zeros((boxes.shape[0],), dtype=torch.long)
    return keep_tensor, num_to_keep, parent_index
