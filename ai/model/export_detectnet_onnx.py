from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

FORBIDDEN_OPS = {
    "GatherND",
    "NonMaxSuppression",
    "TopK",
    "Loop",
    "If",
    "Scan",
    "RoiAlign",
}


def import_module_from_path(py_path: Path):
    py_path = py_path.resolve()
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from: {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[py_path.stem] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description="Export detectNet-compatible ONNX for JetPack 4.6 / TensorRT 8.0.1")
    p.add_argument("--model-py", type=str, default="model_export.py")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--labels", type=str, required=True, help="training labels.txt without BACKGROUND line")
    p.add_argument("--output", type=str, default="ssd-mobilenet.onnx")
    p.add_argument("--deploy-labels", type=str, default="labels_detectnet.txt", help="output labels file for detectnet (BACKGROUND prepended)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--input-name", type=str, default="input_0")
    p.add_argument("--scores-name", type=str, default="scores")
    p.add_argument("--boxes-name", type=str, default="boxes")
    p.add_argument("--opset", type=int, default=13, choices=[13])
    p.add_argument("--no-softmax", action="store_true")
    return p.parse_args()


def read_labels(path: Path) -> list[str]:
    labels = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not labels:
        raise ValueError("labels.txt is empty")
    return labels


def write_detectnet_labels(path: Path, train_labels: list[str]) -> list[str]:
    deploy_labels = ["BACKGROUND"] + list(train_labels)
    path.write_text("\n".join(deploy_labels) + "\n", encoding="utf-8")
    return deploy_labels


def check_onnx(
    onnx_path: Path,
    input_name: str,
    scores_name: str,
    boxes_name: str,
    expected_opset: int,
):
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    opsets = {op.domain: op.version for op in model.opset_import}
    actual_opset = opsets.get("", None)

    if actual_opset != expected_opset:
        raise RuntimeError(f"unexpected opset: {actual_opset} (expected {expected_opset})")

    op_types = [node.op_type for node in model.graph.node]
    bad = sorted({op for op in op_types if op in FORBIDDEN_OPS})
    if bad:
        raise RuntimeError("forbidden ONNX ops detected: " + ", ".join(bad))

    names_in = [x.name for x in model.graph.input]
    names_out = [x.name for x in model.graph.output]
    if names_in != [input_name]:
        raise RuntimeError(f"unexpected input names: {names_in}")
    if names_out != [scores_name, boxes_name]:
        raise RuntimeError(f"unexpected output names: {names_out}")

    def dims_of(v):
        dims = []
        for d in v.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                dims.append(str(d.dim_param))
            else:
                dims.append("?")
        return dims

    return {
        "opset": actual_opset,
        "input": dims_of(model.graph.input[0]),
        "scores": dims_of(model.graph.output[0]),
        "boxes": dims_of(model.graph.output[1]),
        "ops": sorted(set(op_types)),
        "producer": f"{model.producer_name} {model.producer_version}",
    }


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    model_py = Path(args.model_py)
    ckpt_path = Path(args.checkpoint)
    labels_path = Path(args.labels)
    out_path = Path(args.output)
    deploy_labels_path = Path(args.deploy_labels)

    module = import_module_from_path(model_py)
    model, ckpt_classes, img_size, _ = module.load_checkpoint_model(str(ckpt_path), device=str(device))

    train_labels = read_labels(labels_path)
    if list(train_labels) != list(ckpt_classes):
        raise ValueError(
            "checkpoint classes and labels.txt mismatch\n"
            f"checkpoint: {ckpt_classes}\n"
            f"labels.txt: {train_labels}"
        )

    deploy_labels = write_detectnet_labels(deploy_labels_path, train_labels)

    wrapper = module.DetectNetONNXExportWrapper(
        model=model,
        img_size=img_size,
        apply_softmax=not args.no_softmax,
    ).to(device).eval()

    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)
    scores, boxes = wrapper(dummy)

    if tuple(scores.shape)[:2] != (1, wrapper.anchors.shape[1]):
        raise RuntimeError(f"unexpected scores shape {tuple(scores.shape)}")
    if tuple(boxes.shape) != (1, wrapper.anchors.shape[1], 4):
        raise RuntimeError(f"unexpected boxes shape {tuple(boxes.shape)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[args.input_name],
        output_names=[args.scores_name, args.boxes_name],
        dynamic_axes=None,
        dynamo=False,          # 추가
        external_data=True,    # 명시
    )

    info = check_onnx(out_path, args.input_name, args.scores_name, args.boxes_name, args.opset)

    print(f"saved ONNX          : {out_path}")
    print(f"saved deploy labels : {deploy_labels_path}")
    print(f"input dims          : {info['input']}")
    print(f"scores dims         : {info['scores']}")
    print(f"boxes dims          : {info['boxes']}")
    print(f"deploy labels       : {deploy_labels}")
    print("forbidden ops check : PASS")
    print("\nRun on Jetson detectnet with:")
    print(
        f"detectnet --model={out_path} --labels={deploy_labels_path} "
        f"--input-blob={args.input_name} --output-cvg={args.scores_name} --output-bbox={args.boxes_name} "
        f"--threshold=0.5 input.jpg output.jpg"
    )


if __name__ == "__main__":
    main()
