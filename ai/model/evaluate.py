from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF


# ============================================================
# Default run config
# ============================================================
DEFAULT_MODEL_PY = "model_detectnet_export_trt801.py"
DEFAULT_CHECKPOINTS = ["checkpoints/best.pth"]
DEFAULT_DATA_ROOT = "dataset"
DEFAULT_SPLIT = "test"
DEFAULT_SCORE_THRESH = 0.05
DEFAULT_IOU_MATCH_THRESH = 0.50
DEFAULT_OUTPUT_DIR = "eval_results"


# ============================================================
# Utils
# ============================================================
def collate_fn(batch):
    return tuple(zip(*batch))


def find_image_path(img_dir: Path, image_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"):
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def read_split_ids(split_file: Path) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")
    return [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_labels_from_txt(label_file: Path) -> List[str]:
    if not label_file.exists():
        raise FileNotFoundError(f"labels.txt not found: {label_file}")
    classes = [x.strip() for x in label_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not classes:
        raise ValueError("labels.txt is empty")
    return classes


def parse_voc_xml(xml_path: Path, class_to_idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    root = ET.parse(xml_path).getroot()
    boxes, labels = [], []

    for obj in root.findall("object"):
        name_tag = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_tag is None or bndbox is None or not (name_tag.text or "").strip():
            continue

        cls_name = name_tag.text.strip()
        if cls_name not in class_to_idx:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_idx[cls_name])

    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)


# ============================================================
# Dataset
# ============================================================
class VOCTestDataset(Dataset):
    def __init__(self, img_dir: Path, ann_dir: Path, image_ids: List[str], class_to_idx: Dict[str, int]):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.image_ids = image_ids
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = find_image_path(self.img_dir, image_id)
        if img_path is None:
            raise FileNotFoundError(f"image not found for id={image_id}")

        xml_path = self.ann_dir / f"{image_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"annotation not found: {xml_path}")

        image = Image.open(img_path).convert("RGB")
        image_tensor = TF.to_tensor(image)

        gt_boxes, gt_labels = parse_voc_xml(xml_path, self.class_to_idx)

        meta = {
            "image_id": image_id,
            "width": image.width,
            "height": image.height,
        }
        target = {"boxes": gt_boxes, "labels": gt_labels}
        return image_tensor, target, meta


# ============================================================
# Model loading
# ============================================================
def load_model_module(model_py_path: Path):
    spec = importlib.util.spec_from_file_location("user_model_module", str(model_py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import model.py from {model_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint_paths(ckpt_args: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in ckpt_args:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pth")))
        elif any(ch in item for ch in "*?[]"):
            paths.extend(sorted(Path().glob(item)))
        else:
            paths.append(p)

    paths = [p.resolve() for p in paths]

    uniq = []
    seen = set()
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    if not uniq:
        raise FileNotFoundError("no checkpoint files found")

    return uniq


# ============================================================
# Evaluation core
# ============================================================
def filter_predictions(pred: Dict[str, torch.Tensor], score_thresh: float) -> Dict[str, torch.Tensor]:
    boxes = pred["boxes"].detach().cpu()
    labels = pred["labels"].detach().cpu()
    scores = pred["scores"].detach().cpu()

    keep = scores >= score_thresh
    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
    }


def greedy_match(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_match_thresh: float,
):
    """
    GT 중심 1:1 greedy matching

    - matched + same class   -> TP
    - matched + wrong class  -> FP 1개 + FN 1개
    - unmatched GT           -> FN
    - unmatched prediction   -> FP
    """
    results = []
    used_pred = set()

    if gt_boxes.numel() == 0:
        return results, used_pred

    if pred_boxes.numel() == 0:
        for gi in range(len(gt_boxes)):
            results.append({
                "gt_idx": gi,
                "pred_idx": None,
                "gt_label": int(gt_labels[gi].item()),
                "pred_label": None,
                "iou": 0.0,
                "score": 0.0,
                "match_type": "missed",
                "is_tp": False,
            })
        return results, used_pred

    iou_mat = box_iou(gt_boxes, pred_boxes)

    candidates = []
    for gi in range(iou_mat.shape[0]):
        for pi in range(iou_mat.shape[1]):
            iou = float(iou_mat[gi, pi].item())
            if iou >= iou_match_thresh:
                same_label_bonus = 1 if int(gt_labels[gi].item()) == int(pred_labels[pi].item()) else 0
                score = float(pred_scores[pi].item())
                candidates.append((same_label_bonus, iou, score, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    for same_label_bonus, iou, score, gi, pi in candidates:
        if gi in matched_gt or pi in used_pred:
            continue

        matched_gt.add(gi)
        used_pred.add(pi)

        gt_label = int(gt_labels[gi].item())
        pred_label = int(pred_labels[pi].item())
        is_tp = gt_label == pred_label

        results.append({
            "gt_idx": gi,
            "pred_idx": pi,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "iou": iou,
            "score": score,
            "match_type": "tp" if is_tp else "wrong_class",
            "is_tp": is_tp,
        })

    for gi in range(len(gt_boxes)):
        if gi not in matched_gt:
            results.append({
                "gt_idx": gi,
                "pred_idx": None,
                "gt_label": int(gt_labels[gi].item()),
                "pred_label": None,
                "iou": 0.0,
                "score": 0.0,
                "match_type": "missed",
                "is_tp": False,
            })

    results.sort(key=lambda x: x["gt_idx"])
    return results, used_pred


def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if len(recalls) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_overall_ap50(
    pred_records_all: List[Dict],
    gt_records_all: Dict[str, torch.Tensor],
    iou_thresh: float = 0.50,
):
    """
    라벨 무시 overall AP@0.5
    """
    total_gt = 0
    gt_used = {}

    for image_id, gt_boxes in gt_records_all.items():
        total_gt += int(gt_boxes.shape[0])
        gt_used[image_id] = np.zeros((gt_boxes.shape[0],), dtype=bool)

    preds_sorted = sorted(pred_records_all, key=lambda x: x["score"], reverse=True)

    tp = np.zeros((len(preds_sorted),), dtype=np.float32)
    fp = np.zeros((len(preds_sorted),), dtype=np.float32)

    for i, pred in enumerate(preds_sorted):
        image_id = pred["image_id"]
        pred_box = pred["box"].unsqueeze(0)

        if image_id not in gt_records_all or gt_records_all[image_id].numel() == 0:
            fp[i] = 1.0
            continue

        gt_boxes = gt_records_all[image_id]
        ious = box_iou(pred_box, gt_boxes)[0]
        best_iou, best_idx = torch.max(ious, dim=0)

        best_iou = float(best_iou.item())
        best_idx = int(best_idx.item())

        if best_iou >= iou_thresh and not gt_used[image_id][best_idx]:
            tp[i] = 1.0
            gt_used[image_id][best_idx] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    if total_gt > 0:
        recalls = cum_tp / total_gt
    else:
        recalls = np.zeros_like(cum_tp)

    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    ap50 = compute_ap_from_pr(recalls, precisions) if total_gt > 0 else 0.0

    return {
        "ap50": float(ap50),
        "gt_count": int(total_gt),
        "num_predictions": int(len(preds_sorted)),
        "recall_curve": recalls.tolist(),
        "precision_curve": precisions.tolist(),
    }


def evaluate_checkpoint(
    model,
    loader: DataLoader,
    idx_to_class: Dict[int, str],
    device: str,
    score_thresh: float,
    iou_match_thresh: float,
):
    num_classes = len(idx_to_class)
    missed_col = num_classes
    cm = np.zeros((num_classes, num_classes + 1), dtype=np.int64)

    per_class = {
        idx: {
            "gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tp_iou_sum": 0.0,
            "tp_count": 0,
        }
        for idx in range(1, num_classes + 1)
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tp_iou_sum = 0.0
    total_tp_iou_count = 0

    image_rows = []

    pred_records_all = []
    gt_records_all = {}

    model.eval()
    with torch.no_grad():
        for images, targets, metas in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target, meta in zip(outputs, targets, metas):
                image_id = meta["image_id"]
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()
                pred = filter_predictions(output, score_thresh=score_thresh)

                # overall AP 계산용 (라벨 무시)
                gt_records_all[image_id] = gt_boxes.clone()
                for box, score in zip(pred["boxes"], pred["scores"]):
                    pred_records_all.append({
                        "image_id": image_id,
                        "box": box.clone(),
                        "score": float(score.item()),
                    })

                results, used_pred = greedy_match(
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    pred_boxes=pred["boxes"],
                    pred_labels=pred["labels"],
                    pred_scores=pred["scores"],
                    iou_match_thresh=iou_match_thresh,
                )

                for gl in gt_labels.tolist():
                    per_class[int(gl)]["gt"] += 1

                for r in results:
                    gt_idx = int(r["gt_label"])
                    row = gt_idx - 1

                    if r["pred_label"] is None:
                        cm[row, missed_col] += 1
                        per_class[gt_idx]["fn"] += 1
                        total_fn += 1
                    else:
                        pred_idx = int(r["pred_label"])
                        col = pred_idx - 1
                        cm[row, col] += 1

                        if r["match_type"] == "tp":
                            per_class[gt_idx]["tp"] += 1
                            per_class[gt_idx]["tp_iou_sum"] += r["iou"]
                            per_class[gt_idx]["tp_count"] += 1

                            total_tp += 1
                            total_tp_iou_sum += r["iou"]
                            total_tp_iou_count += 1
                        else:
                            per_class[gt_idx]["fn"] += 1
                            per_class[pred_idx]["fp"] += 1
                            total_fn += 1
                            total_fp += 1

                    image_rows.append({
                        "image_id": image_id,
                        "gt_label": idx_to_class[int(r["gt_label"])],
                        "pred_label": idx_to_class[int(r["pred_label"])] if r["pred_label"] is not None else "(missed)",
                        "iou": float(r["iou"]),
                        "score": float(r["score"]),
                        "match_type": r["match_type"],
                        "is_tp": int(r["is_tp"]),
                    })

                for pi, pred_label in enumerate(pred["labels"].tolist()):
                    if pi not in used_pred:
                        pred_idx = int(pred_label)
                        per_class[pred_idx]["fp"] += 1
                        total_fp += 1

                        image_rows.append({
                            "image_id": image_id,
                            "gt_label": "(none)",
                            "pred_label": idx_to_class[pred_idx],
                            "iou": 0.0,
                            "score": float(pred["scores"][pi].item()),
                            "match_type": "extra_fp",
                            "is_tp": 0,
                        })

    per_class_rows = []
    for idx in range(1, num_classes + 1):
        stat = per_class[idx]

        gt = stat["gt"]
        tp = stat["tp"]
        fp = stat["fp"]
        fn = stat["fn"]

        recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
        mean_iou_tp = stat["tp_iou_sum"] / stat["tp_count"] if stat["tp_count"] > 0 else math.nan

        per_class_rows.append({
            "class": idx_to_class[idx],
            "gt_count": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": recall,
            "precision": precision,
            "mean_iou_tp_only": mean_iou_tp,
        })

    overall_ap_result = compute_overall_ap50(
        pred_records_all=pred_records_all,
        gt_records_all=gt_records_all,
        iou_thresh=iou_match_thresh,
    )

    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_iou = total_tp_iou_sum / total_tp_iou_count if total_tp_iou_count > 0 else 0.0

    summary = {
        "recall": float(overall_recall),
        "precision": float(overall_precision),
        "mean_iou": float(overall_iou),
        "overall_ap50": float(overall_ap_result["ap50"]),
        "total_gt": int(sum(r["gt_count"] for r in per_class_rows)),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
    }

    return {
        "summary": summary,
        "per_class_rows": per_class_rows,
        "overall_ap": overall_ap_result,
        "confusion_matrix": cm,
        "image_rows": image_rows,
    }


# ============================================================
# Output helpers
# ============================================================
def save_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_confusion_matrix_png(cm: np.ndarray, class_names: List[str], save_path: Path, title: str):
    pred_names = class_names + ["(missed)"]
    fig_w = max(8, len(pred_names) * 1.2)
    fig_h = max(6, len(class_names) * 0.8)

    n_rows, n_cols = cm.shape
    missed_col = n_cols - 1

    color_img = np.ones((n_rows, n_cols, 3), dtype=np.float32)

    max_val = int(cm.max()) if cm.size > 0 else 0
    if max_val == 0:
        max_val = 1

    GREEN = np.array([0.10, 0.65, 0.10], dtype=np.float32)
    ORANGE = np.array([1.00, 0.60, 0.00], dtype=np.float32)
    RED = np.array([0.85, 0.15, 0.15], dtype=np.float32)
    WHITE = np.array([1.00, 1.00, 1.00], dtype=np.float32)

    def blend_with_white(target_rgb, value, vmax):
        norm = (value / vmax) ** 0.7
        return WHITE * (1 - norm) + target_rgb * norm

    for r in range(n_rows):
        for c in range(n_cols):
            val = int(cm[r, c])

            if val == 0:
                color_img[r, c] = WHITE
            else:
                if c == missed_col:
                    color_img[r, c] = blend_with_white(RED, val, max_val)
                elif r == c:
                    color_img[r, c] = blend_with_white(GREEN, val, max_val)
                else:
                    color_img[r, c] = blend_with_white(ORANGE, val, max_val)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(color_img, aspect="auto")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(n_rows):
        for c in range(n_cols):
            val = int(cm[r, c])
            cell_rgb = color_img[r, c]
            luminance = 0.299 * cell_rgb[0] + 0.587 * cell_rgb[1] + 0.114 * cell_rgb[2]
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(c, r, str(val), ha="center", va="center", fontsize=10, color=text_color)

    ax.set_xticks(range(len(pred_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(pred_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Pred Value")
    ax.set_ylabel("True Value")
    ax.set_title(title)

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=GREEN, label="True = Pred"),
        mpatches.Patch(color=ORANGE, label="True ≠ Pred"),
        mpatches.Patch(color=RED, label="Pred = missed"),
        mpatches.Patch(color=WHITE, label="Value = 0"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_recall_iou_bar_png(per_class_rows: List[Dict], save_path: Path, title: str):
    classes = [r["class"] for r in per_class_rows]
    recall_vals = [0.0 if math.isnan(r["recall"]) else r["recall"] for r in per_class_rows]
    iou_vals = [0.0 if math.isnan(r["mean_iou_tp_only"]) else r["mean_iou_tp_only"] for r in per_class_rows]

    x = np.arange(len(classes))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.8), 6))
    b1 = ax.bar(x - w / 2, recall_vals, width=w, label="Recall")
    b2 = ax.bar(x + w / 2, iou_vals, width=w, label="Mean IoU (TP only)")

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_overall_ap_bar_png(ap50: float, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 6))
    bars = ax.bar(["Overall AP@0.5"], [ap50])

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("AP")
    ax.set_title(title)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_overall_pr_curve_png(overall_ap: Dict, save_path: Path, title: str):
    recall = overall_ap["recall_curve"]
    precision = overall_ap["precision_curve"]
    ap50 = overall_ap["ap50"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(recall) == 0:
        ax.plot([0], [0], label=f"Overall (AP={ap50:.3f})")
    else:
        ax.plot(recall, precision, label=f"Overall (AP={ap50:.3f})")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def print_default_run_guide():
    print("=" * 80)
    print("Using built-in defaults")
    print(f"  model.py         : {DEFAULT_MODEL_PY}")
    print(f"  checkpoints      : {DEFAULT_CHECKPOINTS}")
    print(f"  data_root        : {DEFAULT_DATA_ROOT}")
    print(f"  split            : {DEFAULT_SPLIT}")
    print(f"  score_thresh     : {DEFAULT_SCORE_THRESH}")
    print(f"  iou_match_thresh : {DEFAULT_IOU_MATCH_THRESH}")
    print(f"  output_dir       : {DEFAULT_OUTPUT_DIR}")
    print("=" * 80)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one or more SSD checkpoints with overall Recall/Precision/IoU/AP")
    p.add_argument("--model-py", type=str, default=DEFAULT_MODEL_PY)
    p.add_argument("--checkpoints", type=str, nargs="+", default=DEFAULT_CHECKPOINTS)
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--score-thresh", type=float, default=DEFAULT_SCORE_THRESH)
    p.add_argument("--iou-match-thresh", type=float, default=DEFAULT_IOU_MATCH_THRESH)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print_default_run_guide()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_module = load_model_module(Path(args.model_py))
    if not hasattr(model_module, "load_checkpoint_model"):
        raise AttributeError("model.py must define load_checkpoint_model(checkpoint_path, device)")

    data_root = Path(args.data_root)
    split_file = data_root / "ImageSets" / "Main" / f"{args.split}.txt"
    img_dir = data_root / "JPEGImages"
    ann_dir = data_root / "Annotations"

    checkpoint_paths = resolve_checkpoint_paths(args.checkpoints)

    txt_label_file = data_root / "labels.txt"
    dataset_classes = load_labels_from_txt(txt_label_file) if txt_label_file.exists() else None

    comparison_rows = []

    for ckpt_path in checkpoint_paths:
        print("=" * 80)
        print(f"Evaluating: {ckpt_path}")

        model, ckpt_classes, img_size, ckpt = model_module.load_checkpoint_model(ckpt_path, device=args.device)
        classes = dataset_classes if dataset_classes is not None else ckpt_classes

        if dataset_classes is not None and list(dataset_classes) != list(ckpt_classes):
            print("[WARN] dataset labels.txt and checkpoint classes order are different.")
            print(f"       dataset   : {dataset_classes}")
            print(f"       checkpoint: {ckpt_classes}")
            print("       evaluation uses dataset labels.txt as ground-truth class order.")

        class_to_idx = {name: i + 1 for i, name in enumerate(classes)}
        idx_to_class = {i + 1: name for i, name in enumerate(classes)}

        image_ids = read_split_ids(split_file)
        dataset = VOCTestDataset(
            img_dir=img_dir,
            ann_dir=ann_dir,
            image_ids=image_ids,
            class_to_idx=class_to_idx,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(args.device == "cuda"),
        )

        result = evaluate_checkpoint(
            model=model,
            loader=loader,
            idx_to_class=idx_to_class,
            device=args.device,
            score_thresh=args.score_thresh,
            iou_match_thresh=args.iou_match_thresh,
        )

        ckpt_name = ckpt_path.stem
        ckpt_out = output_dir / ckpt_name
        ckpt_out.mkdir(parents=True, exist_ok=True)

        save_csv(ckpt_out / "per_class_metrics.csv", result["per_class_rows"])
        save_csv(ckpt_out / "image_level_matches.csv", result["image_rows"])
        save_csv(ckpt_out / "overall_ap50.csv", [{
            "overall_ap50": result["overall_ap"]["ap50"],
            "gt_count": result["overall_ap"]["gt_count"],
            "num_predictions": result["overall_ap"]["num_predictions"],
        }])

        draw_confusion_matrix_png(
            cm=result["confusion_matrix"],
            class_names=classes,
            save_path=ckpt_out / "confusion_matrix.png",
            title=f"{ckpt_name} | Confusion Matrix",
        )
        draw_recall_iou_bar_png(
            per_class_rows=result["per_class_rows"],
            save_path=ckpt_out / "recall_iou_bar.png",
            title=f"{ckpt_name} | Recall / IoU by class",
        )
        draw_overall_ap_bar_png(
            ap50=result["overall_ap"]["ap50"],
            save_path=ckpt_out / "overall_ap_bar.png",
            title=f"{ckpt_name} | Overall AP@0.5",
        )
        draw_overall_pr_curve_png(
            overall_ap=result["overall_ap"],
            save_path=ckpt_out / "overall_pr_curve.png",
            title=f"{ckpt_name} | Overall PR Curve @ IoU {args.iou_match_thresh:.2f}",
        )

        s = result["summary"]
        comparison_rows.append({
            "checkpoint": str(ckpt_path),
            "epoch": ckpt.get("epoch", ""),
            "img_size": img_size,
            "recall": s["recall"],
            "precision": s["precision"],
            "mean_iou": s["mean_iou"],
            "overall_ap50": s["overall_ap50"],
            "total_gt": s["total_gt"],
            "total_tp": s["total_tp"],
            "total_fp": s["total_fp"],
            "total_fn": s["total_fn"],
            "score_thresh": args.score_thresh,
            "iou_match_thresh": args.iou_match_thresh,
        })

        print(f"recall        : {s['recall']:.4f}")
        print(f"precision     : {s['precision']:.4f}")
        print(f"mean_iou      : {s['mean_iou']:.4f}")
        print(f"overall_ap50  : {s['overall_ap50']:.4f}")
        print(f"saved to      : {ckpt_out}")

    comparison_rows.sort(
        key=lambda x: (x["recall"], x["precision"], x["overall_ap50"], x["mean_iou"]),
        reverse=True,
    )

    for rank, row in enumerate(comparison_rows, start=1):
        row["rank"] = rank

    ordered_cols = [
        "rank",
        "checkpoint",
        "epoch",
        "img_size",
        "recall",
        "precision",
        "mean_iou",
        "overall_ap50",
        "total_gt",
        "total_tp",
        "total_fp",
        "total_fn",
        "score_thresh",
        "iou_match_thresh",
    ]
    comparison_rows = [{k: row.get(k, "") for k in ordered_cols} for row in comparison_rows]
    save_csv(output_dir / "comparison_summary.csv", comparison_rows)

    print("\n" + "=" * 80)
    print("Final ranking (overall Recall priority)")
    print("=" * 80)
    for row in comparison_rows:
        crown = "👑" if row["rank"] == 1 else f" {row['rank']}"
        print(
            f"{crown} | {Path(row['checkpoint']).name:20s} "
            f"recall={row['recall']:.4f} "
            f"precision={row['precision']:.4f} "
            f"overall_ap50={row['overall_ap50']:.4f} "
            f"mean_iou={row['mean_iou']:.4f}"
        )

    print(f"\ncomparison csv: {output_dir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()