# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys; sys.path.insert(0, ".")

from det3d.models.backbones.scn import SpMiddleResNetFHD
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint
import torch
import pickle
import argparse
from torch import nn
# custom functional package
from export_voxel_onnx import funcs
from export_voxel_onnx import exptool
import os
import onnx
from onnxsim import simplify
import onnxoptimizer
import struct
class center_voxel_detection(nn.Module):
    def __init__(self, model):
        super(center_voxel_detection, self).__init__()
        self.with_neck = model.with_neck
        self.neck = model.neck
        self.bbox_head = model.bbox_head

    def forward(self, x):
        if self.with_neck:
            x = self.neck(x)
        preds = self.bbox_head(x)[0]
        # for task in range(len(preds)):
        #     hm_preds = torch.sigmoid(preds[task]['hm'])
        #     preds[task]['dim'] = torch.exp(preds[task]['dim'])
        #     scores, labels = torch.max(hm_preds, dim=1)
        #     preds[task]["hm"] = (scores, labels)
        return preds

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    parser.add_argument("--in-channel", type=int, default=5, help="SCN num of input channels")
    parser.add_argument("--config", type=str, default='configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py', help="voxel training config")
    parser.add_argument("--ckpt", type=str, default="./checkpoint/waymo_centerpoint_voxelnet_3x/last_train.pth", help="SCN Checkpoint (scn backbone checkpoint)")
    parser.add_argument("--input", type=str, default=None, help="input pickle data, random if there have no input")
    parser.add_argument("--save-folder", type=str, default="./onnx_model", help="output onnx")
    parser.add_argument("--save-pfe-onnx", type=str, default="pfe_voxel_waymo_zf_2000_frame.onnx", help="output onnx")
    parser.add_argument("--save-rpn-onnx", type=str, default="rpn_voxel_waymo_zf_2000_frame.onnx", help="output onnx")
    parser.add_argument("--save-tensor", type=str, default=None, help="Save input/output tensor to file. The purpose of this operation is to verify the inference result of c++")
    args = parser.parse_args()

    model = SpMiddleResNetFHD(args.in_channel).cuda().eval().half()
    if args.ckpt:
        model = funcs.load_scn_backbone_checkpoint(model, args.ckpt)

    model = funcs.layer_fusion(model)

    print("Fusion model:")
    print(model)

    if args.input:
        with open(args.input, "rb") as f:
            voxels, coors, spatial_shape, batch_size = pickle.load(f)
            voxels = torch.tensor(voxels).half().cuda()
            coors  = torch.tensor(coors).int().cuda()
    else:
        voxels = torch.zeros(1, args.in_channel).half().cuda()
        coors  = torch.zeros(1, 4).int().cuda()
        batch_size    = 1
        spatial_shape = [2400, 2400, 40]

    pfe_out = model(voxels, coors, batch_size, spatial_shape)
    exptool.export_onnx(model, voxels, coors, batch_size, spatial_shape, os.path.join(args.save_folder, args.save_pfe_onnx), args.save_tensor)
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.ckpt is not None:
        load_checkpoint(model, args.ckpt, map_location="cpu")
    print("cast model to fp16")
    model = model.half()
    model = model.cuda()
    model.eval()
    rpn_model = center_voxel_detection(model)
    output_names = ['hm_0', 'rot_0', 'dim_0', 'height_0', 'reg_0']
    output_names.reverse()
    torch.onnx.export(rpn_model,
                      pfe_out,
                      os.path.join(args.save_folder, args.save_rpn_onnx),
                      output_names=output_names,
                      opset_version=14)
    rpn_model = onnx.load(os.path.join(args.save_folder, args.save_rpn_onnx))
    rpn_model, check = simplify(rpn_model)
    for i, node in enumerate(rpn_model.graph.node):
        if node.op_type == 'Pad':
            pads = node.input[1]
            for j, data in enumerate(rpn_model.graph.initializer):
                if data.name == pads:
                    fmt = 'L' * data.dims[0]
                    pads = struct.unpack(fmt, data.raw_data)
            for k, onode in enumerate(rpn_model.graph.node):
                for m, input in enumerate(onode.input):
                    if input == node.output[0]:
                        for n, a in enumerate(onode.attribute):
                            if a.name == 'pads':
                                rpn_model.graph.node[k].attribute[n].ints[0] = pads[2]
                                rpn_model.graph.node[k].attribute[n].ints[1] = pads[3]
                                rpn_model.graph.node[k].attribute[n].ints[2] = pads[6]
                                rpn_model.graph.node[k].attribute[n].ints[3] = pads[7]
                        rpn_model.graph.node[k].input[0] = node.input[0]

    rpn_model, check = simplify(rpn_model)

    name = 'input'
    for i, node in enumerate(rpn_model.graph.node):
        for j, input in enumerate(node.input):
            if input == rpn_model.graph.input[0].name:
                rpn_model.graph.node[i].input[j] = name
    rpn_model.graph.input[0].name = name
    onnx.save(rpn_model, os.path.join(args.save_folder, args.save_rpn_onnx))
    print()