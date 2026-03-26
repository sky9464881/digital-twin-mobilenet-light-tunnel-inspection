from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class SSDMobileNetV2Backbone(nn.Module):
    def __init__(self, img_size: int = 320, pretrained: bool = True, trainable: bool = True):
        super().__init__()
        self.img_size = int(img_size)
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights).features

        self.stage1 = nn.Sequential(*backbone[:7])
        self.stage2 = nn.Sequential(*backbone[7:14])
        self.stage3 = nn.Sequential(*backbone[14:])

        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

        self.extra1 = nn.Sequential(
            ConvBNReLU(1280, 512, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(512, 512, kernel_size=3, stride=2, padding=1),
        )
        self.extra2 = nn.Sequential(
            ConvBNReLU(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(256, 256, kernel_size=3, stride=2, padding=1),
        )
        self.extra3 = nn.Sequential(
            ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(256, 256, kernel_size=3, stride=2, padding=1),
        )
        self.out_channels = self._get_out_channels()

    def _get_out_channels(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, self.img_size, self.img_size)
            feats = self.forward(x)
            return [f.shape[1] for f in feats.values()]

    def forward(self, x: torch.Tensor):
        out = OrderedDict()
        x = self.stage1(x)
        out["0"] = x
        x = self.stage2(x)
        out["1"] = x
        x = self.stage3(x)
        out["2"] = x
        x = self.extra1(x)
        out["3"] = x
        x = self.extra2(x)
        out["4"] = x
        x = self.extra3(x)
        out["5"] = x
        return out


def build_model(
    num_classes: int,
    img_size: int = 320,
    pretrained_backbone: bool = True,
    trainable_backbone: bool = True,
    score_thresh: float = 0.01,
    nms_thresh: float = 0.45,
    detections_per_img: int = 200,
    topk_candidates: int = 400,
) -> SSD:
    backbone = SSDMobileNetV2Backbone(
        img_size=img_size,
        pretrained=pretrained_backbone,
        trainable=trainable_backbone,
    )
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        min_ratio=0.15,
        max_ratio=0.90,
    )
    return SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(img_size, img_size),
        num_classes=num_classes,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
        topk_candidates=topk_candidates,
    )


def load_checkpoint_model(checkpoint_path: str | Path, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    img_size = int(ckpt.get("img_size", 320))

    model = build_model(
        num_classes=len(classes) + 1,
        img_size=img_size,
        pretrained_backbone=False,
        trainable_backbone=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, classes, img_size, ckpt


class DetectNetONNXExportWrapper(nn.Module):
    """
    detectNet ONNX path expects:
      scores: [1, N, C]   (class 0 = BACKGROUND)
      boxes : [1, N, 4]   normalized xyxy in [0,1]

    This wrapper exports exactly that, while keeping TensorRT 8.0.1-safe ops.
    Anchors are precomputed as a constant buffer to keep the graph simple.
    """

    def __init__(self, model: SSD, img_size: int, apply_softmax: bool = True):
        super().__init__()
        self.model = model.eval()
        self.img_size = int(img_size)
        self.apply_softmax = bool(apply_softmax)

        anchors = self._build_static_anchors()
        self.register_buffer("anchors", anchors, persistent=True)  # [1, N, 4]

    @torch.no_grad()
    def _build_static_anchors(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        x = torch.zeros(1, 3, self.img_size, self.img_size, device=device)
        features = self.model.backbone(x)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())
        image_sizes = [(self.img_size, self.img_size)]
        anchors = self.model.anchor_generator(ImageList(x, image_sizes), feature_list)
        anchors = torch.stack(anchors, dim=0)  # [1, N, 4]
        return anchors.detach()

    def _decode_boxes(self, rel_codes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        boxes = anchors.to(dtype=rel_codes.dtype)

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.model.box_coder.weights
        dx = rel_codes[..., 0:1] / wx
        dy = rel_codes[..., 1:2] / wy
        dw = torch.clamp(rel_codes[..., 2:3] / ww, max=self.model.box_coder.bbox_xform_clip)
        dh = torch.clamp(rel_codes[..., 3:4] / wh, max=self.model.box_coder.bbox_xform_clip)

        pred_ctr_x = dx * widths.unsqueeze(-1) + ctr_x.unsqueeze(-1)
        pred_ctr_y = dy * heights.unsqueeze(-1) + ctr_y.unsqueeze(-1)
        pred_w = torch.exp(dw) * widths.unsqueeze(-1)
        pred_h = torch.exp(dh) * heights.unsqueeze(-1)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        boxes = torch.cat((x1, y1, x2, y2), dim=-1)
        boxes = boxes.clamp(min=0.0, max=float(self.img_size))
        scale = boxes.new_tensor([float(self.img_size), float(self.img_size), float(self.img_size), float(self.img_size)]).view(1, 1, 4)
        return boxes / scale

    def forward(self, x: torch.Tensor):
        features = self.model.backbone(x)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())

        head_outputs = self.model.head(feature_list)
        cls_logits = head_outputs["cls_logits"]
        bbox_reg = head_outputs["bbox_regression"]

        scores = torch.softmax(cls_logits, dim=-1) if self.apply_softmax else cls_logits
        boxes = self._decode_boxes(bbox_reg, self.anchors)
        return scores, boxes
