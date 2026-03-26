import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

CLASS_MAP = {
    "scratch": 0,
    "stain": 1,
    "dent": 2,
    "smash": 3,
}

def voc_xml_to_yolo_txt_from_root(root, yolo_txt_path):
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[name]

        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        x_c = ((xmin + xmax) / 2.0) / W
        y_c = ((ymin + ymax) / 2.0) / H
        w   = (xmax - xmin) / W
        h   = (ymax - ymin) / H

        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    yolo_txt_path.write_text("\n".join(lines), encoding="utf-8")


def make_yolo_gray_set(voc_xml_path, gray_dir, yolo_dir):
    voc_xml_path = Path(voc_xml_path)
    gray_dir = Path(gray_dir)
    yolo_dir = Path(yolo_dir)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    base_name = voc_xml_path.stem  # ex) tincase_0000

    # 원본 XML 파싱
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()

    # L/M/R용 파일명
    suffixes = ["L", "M", "R"]
    for s in suffixes:
        img_name = f"{base_name}_{s}.png"

        src_img = gray_dir / img_name
        if not src_img.is_file():
            continue  # 해당 gray 이미지가 없으면 스킵

        # YOLO용 이미지 복사
        dst_img = yolo_dir / img_name
        shutil.copy2(src_img, dst_img)

        # YOLO 라벨 생성 (원본 XML 내용 그대로 사용)
        yolo_txt_path = yolo_dir / f"{base_name}_{s}.txt"
        voc_xml_to_yolo_txt_from_root(root, yolo_txt_path)


# 예시 사용
if __name__ == "__main__":
    voc_dir   = Path(r"C:\Users\hwapyeong\Documents\computer_vision\lighting_tunnel\extract_defect_2\data\260319_rendering_data\annotations")
    gray_dir  = Path(r"C:\Users\hwapyeong\Documents\computer_vision\lighting_tunnel\extract_defect_2\data\260319_rendering_data\gray")
    yolo_dir  = Path(r"C:\Users\hwapyeong\Documents\computer_vision\lighting_tunnel\extract_defect_2\data\260319_rendering_data\YOLO")

    for xml in voc_dir.glob("tincase_*.xml"):
        make_yolo_gray_set(xml, gray_dir, yolo_dir)
