from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD
from torchvision.transforms import functional as TF
from tqdm import tqdm


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_labels(label_file: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_file = Path(label_file)
    if not label_file.exists():
        raise FileNotFoundError(f"labels.txt not found: {label_file}")

    classes = [x.strip() for x in label_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not classes:
        raise ValueError("labels.txt is empty")

    class_to_idx = {name: i + 1 for i, name in enumerate(classes)}  # 0 is background
    idx_to_class = {i + 1: name for i, name in enumerate(classes)}
    return classes, class_to_idx, idx_to_class


def read_split_ids(split_file: Path) -> List[str]:
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")
    return [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]


def find_image_path(img_dir: Path, image_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"):
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def validate_dataset(img_dir: Path, ann_dir: Path, split_ids: List[str], class_to_idx: Dict[str, int], split_name: str) -> None:
    missing_images, missing_xmls, unknown_labels, invalid_boxes = [], [], [], []

    for image_id in split_ids:
        img_path = find_image_path(img_dir, image_id)
        xml_path = ann_dir / f"{image_id}.xml"

        if img_path is None:
            missing_images.append(image_id)
        if not xml_path.exists():
            missing_xmls.append(image_id)
            continue

        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_tag is None or not (name_tag.text or "").strip():
                continue

            cls_name = name_tag.text.strip()
            if cls_name not in class_to_idx:
                unknown_labels.append((image_id, cls_name))

            if bndbox is not None:
                try:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    if xmax <= xmin or ymax <= ymin:
                        invalid_boxes.append((image_id, xmin, ymin, xmax, ymax))
                except Exception:
                    invalid_boxes.append((image_id, "parse_error"))

    errors = []
    if missing_images:
        errors.append(f"[{split_name}] missing images: {missing_images[:10]}")
    if missing_xmls:
        errors.append(f"[{split_name}] missing xml: {missing_xmls[:10]}")
    if unknown_labels:
        errors.append(f"[{split_name}] unknown labels: {unknown_labels[:10]}")
    if invalid_boxes:
        errors.append(f"[{split_name}] invalid boxes: {invalid_boxes[:10]}")

    if errors:
        raise ValueError("\n".join(errors))

    print(f"[OK] {split_name}: {len(split_ids)} samples validated")


# ============================================================
# Dataset
# ============================================================
class VOCDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        ann_dir: Path,
        split_ids: List[str],
        class_to_idx: Dict[str, int],
        train: bool = True,
        hflip_prob: float = 0.5,
    ):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.image_ids = split_ids
        self.class_to_idx = class_to_idx
        self.train = train
        self.hflip_prob = float(hflip_prob)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _parse_xml(self, xml_path: Path):
        root = ET.parse(xml_path).getroot()
        boxes, labels, iscrowd = [], [], []

        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_tag is None or not (name_tag.text or "").strip() or bndbox is None:
                continue

            cls_name = name_tag.text.strip()
            if cls_name not in self.class_to_idx:
                continue

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])
            iscrowd.append(0)

        return boxes, labels, iscrowd

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = find_image_path(self.img_dir, image_id)
        if img_path is None:
            raise FileNotFoundError(f"image not found for id={image_id}")

        xml_path = self.ann_dir / f"{image_id}.xml"
        image = Image.open(img_path).convert("RGB")
        width, _ = image.size

        boxes, labels, iscrowd = self._parse_xml(xml_path)
        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
            area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)

        if self.train and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            if boxes_t.numel() > 0:
                x1 = width - boxes_t[:, 2]
                x2 = width - boxes_t[:, 0]
                boxes_t[:, 0] = x1
                boxes_t[:, 2] = x2

        image_t = TF.to_tensor(image)
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": area_t,
            "iscrowd": iscrowd_t,
        }
        return image_t, target


# ============================================================
# Model
# ============================================================
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


# ============================================================
# detectNet export wrapper
# ============================================================
class DetectNetONNXExportWrapper(nn.Module):
    """
    Export only two outputs for detectNet:
      scores: [B, N, C]
      boxes : [B, N, 4] normalized xyxy in [0,1]

    No NMS, no TopK, no GatherND.
    Post-processing is left to jetson-inference / detectNet.
    """

    def __init__(self, model: SSD, apply_softmax: bool = True, clamp_boxes: bool = True):
        super().__init__()
        self.model = model
        self.apply_softmax = bool(apply_softmax)
        self.clamp_boxes = bool(clamp_boxes)

    def _decode_boxes(self, rel_codes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        boxes = anchors.to(rel_codes.dtype)
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.model.box_coder.weights
        dx = rel_codes[..., 0] / wx
        dy = rel_codes[..., 1] / wy
        dw = torch.clamp(rel_codes[..., 2] / ww, max=self.model.box_coder.bbox_xform_clip)
        dh = torch.clamp(rel_codes[..., 3] / wh, max=self.model.box_coder.bbox_xform_clip)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _clip_and_normalize(self, boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if self.clamp_boxes:
            boxes_x = boxes[..., [0, 2]].clamp(min=0.0, max=float(width))
            boxes_y = boxes[..., [1, 3]].clamp(min=0.0, max=float(height))
            boxes = torch.stack((boxes_x[..., 0], boxes_y[..., 0], boxes_x[..., 1], boxes_y[..., 1]), dim=-1)

        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
        return boxes / scale

    def forward(self, x: torch.Tensor):
        features = self.model.backbone(x)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())

        head_outputs = self.model.head(feature_list)
        image_h = int(x.shape[-2])
        image_w = int(x.shape[-1])
        image_sizes = [(image_h, image_w) for _ in range(int(x.shape[0]))]
        anchors = self.model.anchor_generator(ImageList(x, image_sizes), feature_list)
        anchors = torch.stack(anchors, dim=0)

        cls_logits = head_outputs["cls_logits"]
        bbox_reg = head_outputs["bbox_regression"]
        scores = torch.softmax(cls_logits, dim=-1) if self.apply_softmax else cls_logits
        boxes = self._decode_boxes(bbox_reg, anchors)
        boxes = self._clip_and_normalize(boxes, image_h, image_w)
        return scores, boxes


# ============================================================
# Training
# ============================================================
class CFG:
    SAVE_DIR = Path("checkpoints")
    NUM_EPOCHS = 20
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    STEP_SIZE = 15
    GAMMA = 0.1
    IMG_SIZE = 320
    HFLIP_PROB = 0.5
    SEED = 42
    SAVE_EVERY = 5


def train_one_epoch(model, loader, optimizer, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch: int) -> float:
    model.train()  # torchvision SSD returns loss only in train mode
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Val   {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())
        total_loss += float(loss.item())
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train SSD-MobileNetV2 and save best.pth for detectNet ONNX export")
    p.add_argument("--data-root", type=str, required=True, help="dataset root containing JPEGImages/ Annotations/ ImageSets/Main/ labels.txt")
    p.add_argument("--save-dir", type=str, default=str(CFG.SAVE_DIR))
    p.add_argument("--epochs", type=int, default=CFG.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=CFG.NUM_WORKERS)
    p.add_argument("--lr", type=float, default=CFG.LR)
    p.add_argument("--weight-decay", type=float, default=CFG.WEIGHT_DECAY)
    p.add_argument("--momentum", type=float, default=CFG.MOMENTUM)
    p.add_argument("--step-size", type=int, default=CFG.STEP_SIZE)
    p.add_argument("--gamma", type=float, default=CFG.GAMMA)
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--img-size", type=int, default=CFG.IMG_SIZE, choices=[320, 640])
    p.add_argument("--hflip-prob", type=float, default=CFG.HFLIP_PROB)
    p.add_argument("--seed", type=int, default=CFG.SEED)
    p.add_argument("--save-every", type=int, default=CFG.SAVE_EVERY)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    img_dir = data_root / "JPEGImages"
    ann_dir = data_root / "Annotations"
    split_dir = data_root / "ImageSets" / "Main"
    label_file = data_root / "labels.txt"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    classes, class_to_idx, _ = load_labels(label_file)
    num_classes = len(classes) + 1

    train_ids = read_split_ids(split_dir / "train.txt")
    val_ids = read_split_ids(split_dir / "val.txt")
    validate_dataset(img_dir, ann_dir, train_ids, class_to_idx, "train")
    validate_dataset(img_dir, ann_dir, val_ids, class_to_idx, "val")

    train_dataset = VOCDataset(img_dir, ann_dir, train_ids, class_to_idx, train=True, hflip_prob=args.hflip_prob)
    val_dataset = VOCDataset(img_dir, ann_dir, val_ids, class_to_idx, train=False, hflip_prob=0.0)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    model = build_model(
        num_classes=num_classes,
        img_size=args.img_size,
        pretrained_backbone=True,
        trainable_backbone=True,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = None if args.no_scheduler else torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_loss = float("inf")
    print(f"classes            : {classes}")
    print(f"num_classes total  : {num_classes} (background included)")
    print(f"device             : {device}")
    print(f"img_size           : {args.img_size}")
    print("normalization      : (x/255 - 0.5)/0.5")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, device, epoch)
        if scheduler is not None:
            scheduler.step()

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": classes,
            "val_loss": val_loss,
            "img_size": args.img_size,
            "normalization": {"range": [0.0, 1.0], "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "detectnet_export": {"input_name": "input_0", "scores_name": "scores", "boxes_name": "boxes"},
        }
        torch.save(ckpt, save_dir / "latest.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, save_dir / "best.pth")

        if epoch % args.save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch}.pth")

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val_loss={best_val_loss:.4f}")

    print(f"Training finished. best checkpoint: {save_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
