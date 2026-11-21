from pathlib import Path
import shutil
import cv2

RAW_DIR = Path("data/VisDrone2019-VID-val")

OUT_DIR = Path("data/yolo_dataset")

for split in ["train", "val"]:
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    
PERSON_CLASSES = {1, 2}

def parse_visdrone_annotation(txt_path: Path) -> dict[int, list[tuple[int, int, int, int]]]:
    frame_boxes = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(",") if x.strip()]
            if len(parts) < 8:
                continue
            frame_idx, _, left, top, w, h, score, category = parts[:8]
            
            if score > 0 and category in PERSON_CLASSES and w > 0 and h > 0:
                frame_boxes.setdefault(frame_idx, []).append((left, top, w, h))
    return frame_boxes

seq_dirs = sorted((RAW_DIR / "sequences").iterdir())
ann_dir = RAW_DIR / "annotations"

for idx, seq_dir in enumerate(seq_dirs):
    if not seq_dir.is_dir():
        continue

    split = "train" if idx < 5 else "val"
    frames_boxes = parse_visdrone_annotation(ann_dir / f"{seq_dir.name}.txt")

    # Беремо кожен 3-й кадр, щоб прибрати дублікати сусідніх кадрів
    images = sorted(seq_dir.glob("*.jpg"))[::3]

    for img_path in images:
        frame_num = int(img_path.stem)  # "0000015.jpg" -> 15
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Унікальне ім'я файлу, щоб кадри з різних відео не перезаписували одне одного
        new_name = f"{seq_dir.name}_{img_path.name}"
        shutil.copy2(img_path, OUT_DIR / "images" / split / new_name)

        # Записуємо розмітку у форматі YOLO (якщо людей немає — створиться порожній .txt як background)
        yolo_lines = []
        for left, top, w, h in frames_boxes.get(frame_num, []):
            x_center = (left + w / 2) / img_w
            y_center = (top + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        label_path = OUT_DIR / "labels" / split / f"{Path(new_name).stem}.txt"
        label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

# Створюємо головний конфіг для навчання YOLOv8
yaml_content = f"""path: {OUT_DIR.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: person
"""
(OUT_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")
print("Датасет успішно підготовлено у:", OUT_DIR)