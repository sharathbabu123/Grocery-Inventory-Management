# detect_pantry.py
# Usage:
#   pip install ultralytics opencv-python pandas
#   python detect_pantry.py --image "/path/to/pantry.jpg" --outdir outputs
#
# What it does:
# - If YOLO-World weights are available, it uses open-vocab prompts like ["jar","packet","pouch","bottle","can","box","tin","spice jar","container"].
# - Otherwise falls back to standard YOLOv8 (COCO classes).
# - Saves: outputs/annotated.jpg and outputs/detections.csv

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

GROCERY_PROMPTS = [
    "jar", "bottle", "can", "tin", "box", "packet", "pouch",
    "plastic container", "steel container", "spice jar",
    "salt", "sugar", "lentils", "dal", "rice", "flour", "oil"
]

COCO_TO_GROCERY_MAP = {
    # crude mapping to make COCO classes more pantry-ish
    "bottle": "bottle",
    "cup": "container",
    "bowl": "container",
    "wine glass": "glass",
    "knife": "utensil",
    "spoon": "utensil",
    "fork": "utensil",
    "banana": "banana",
    "apple": "apple",
    "orange": "orange",
    "sandwich": "food",
    "broccoli": "vegetable",
    "carrot": "vegetable",
    "donut": "snack",
    "cake": "snack",
    "book": "box_or_pack",  # often misfires as "book" for packets/boxes on shelves
    "cell phone": "pack",
}

def try_load_yoloworld():
    """
    Tries to load YOLO-World weights (open-vocabulary).
    If not available locally, returns None (the script will fall back to YOLOv8).
    Common weights: 'yolov8x-world.pt' (Ultralytics).
    """
    # You can place yolov8x-world.pt in the current dir or give a full path here
    for w in ["yolov8x-world.pt", "yolov8l-world.pt", "yolov8m-world.pt"]:
        if Path(w).exists():
            return YOLO(w)
    return None

def load_model():
    m = try_load_yoloworld()
    if m is not None:
        return m, "world"
    # Fallback to standard YOLOv8 (downloads weights automatically on first run)
    return YOLO("yolov8x.pt"), "coco"

def color(i):
    rng = np.random.default_rng(i)
    return (int(rng.integers(50, 255)), int(rng.integers(50, 255)), int(rng.integers(50, 255)))

def cluster_by_shelf(dets, img_h, bands=5):
    """Assign a 'shelf_id' by slicing the image height into horizontal bands."""
    if not dets:
        return dets
    band_h = img_h / bands
    for d in dets:
        y1, y2 = d["ymin"], d["ymax"]
        cy = (y1 + y2) / 2
        shelf = int(np.clip(cy // band_h, 0, bands - 1))
        d["shelf_id"] = shelf
    return dets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Path to pantry image")
    ap.add_argument("--video", help="Path to video file or webcam index")
    ap.add_argument("--outdir", default="outputs", help="Where to save results (image mode)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference size")
    args = ap.parse_args()

    if not args.image and args.video is None:
        ap.error("either --image or --video required")

    model, mode = load_model()
    if mode == "world":
        model.set_classes(GROCERY_PROMPTS)

    if args.image:
        os.makedirs(args.outdir, exist_ok=True)
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        H, W = img.shape[:2]

        results = model.predict(
            source=args.image,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False
        )

        det_list = []
        annotated = img.copy()

        for r in results:
            if r.boxes is None:
                continue
            for i, b in enumerate(r.boxes):
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                if mode == "world":
                    label = r.names.get(cls_id, f"class_{cls_id}")
                else:
                    raw = r.names.get(cls_id, f"class_{cls_id}")
                    label = COCO_TO_GROCERY_MAP.get(raw, raw)

                det_list.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                    "width": x2 - x1, "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1)
                })

        det_list = cluster_by_shelf(det_list, H, bands=5)

        # Draw
        for i, d in enumerate(det_list):
            x1, y1, x2, y2 = d["xmin"], d["ymin"], d["xmax"], d["ymax"]
            c = color(i)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), c, 2)
            txt = f'{d["label"]} {d["confidence"]:.2f} (shelf {d["shelf_id"]})'
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + 8 * len(txt), y1), c, -1)
            cv2.putText(annotated, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

        # Save annotated image & CSV
        out_img = os.path.join(args.outdir, "annotated.jpg")
        out_csv = os.path.join(args.outdir, "detections.csv")
        cv2.imwrite(out_img, annotated)
        pd.DataFrame(det_list).to_csv(out_csv, index=False)

        # Quick shelf-level summary
        if det_list:
            df = pd.DataFrame(det_list)
            shelf_summary = (
                df.groupby(["shelf_id", "label"])
                  .size()
                  .reset_index(name="count")
                  .sort_values(["shelf_id", "count"], ascending=[True, False])
            )
            shelf_summary.to_csv(os.path.join(args.outdir, "shelf_summary.csv"), index=False)
        print(f"[✓] Saved: {out_img}")
        print(f"[✓] Saved: {out_csv}")
        if det_list:
            print(f"[i] Also wrote shelf_summary.csv")
    else:  # video mode
        src = int(args.video) if args.video.isdigit() else args.video
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {args.video}")
        summary = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            H, W = frame.shape[:2]
            annotated = frame.copy()
            det_list = []
            for r in model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False, stream=True):
                if r.boxes is None:
                    continue
                for i, b in enumerate(r.boxes):
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    if mode == "world":
                        label = r.names.get(cls_id, f"class_{cls_id}")
                    else:
                        raw = r.names.get(cls_id, f"class_{cls_id}")
                        label = COCO_TO_GROCERY_MAP.get(raw, raw)
                    det_list.append({
                        "label": label,
                        "confidence": round(conf, 3),
                        "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2
                    })
            det_list = cluster_by_shelf(det_list, H, bands=5)
            for i, d in enumerate(det_list):
                x1, y1, x2, y2 = d["xmin"], d["ymin"], d["xmax"], d["ymax"]
                c = color(i)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), c, 2)
                txt = f'{d["label"]} {d["confidence"]:.2f}'
                cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + 8 * len(txt), y1), c, -1)
                cv2.putText(annotated, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,20,20),1, cv2.LINE_AA)
                summary[d["label"]] = summary.get(d["label"], 0) + 1
            cv2.imshow("pantry", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if summary:
            print("[summary] detections:")
            for k, v in summary.items():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
