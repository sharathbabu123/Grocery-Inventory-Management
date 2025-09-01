# pantry_detect_and_label.py
# Detect pantry containers and recognize item types (dal/rice/sugar/atta/etc.)
# Outputs: annotated.jpg, items.csv, crops/

import argparse
import os
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from ultralytics import YOLO

# ---------- Keyword dictionary (extend freely) ----------
PANTRY_KEYWORDS = {
    "toor_dal":   ["toor dal", "tur dal", "tuvar dal", "arhar dal"],
    "chana_dal":  ["chana dal", "bengal gram dal"],
    "moong_dal":  ["moong dal", "mung dal", "green gram dal"],
    "urad_dal":   ["urad dal", "black gram dal"],
    "masoor_dal": ["masoor dal", "red lentil"],
    "rice":       ["rice", "basmati", "sona masoori", "ponni", "idly rice", "idli rice"],
    "atta":       ["atta", "wheat flour", "chakki atta", "maida", "flour"],
    "sugar":      ["sugar", "shakkar"],
    "salt":       ["salt", "iodized salt", "table salt"],
    "poha":       ["poha", "flattened rice", "aval", "chira"],
    "rava":       ["rava", "sooji", "semolina", "suji"],
    "besan":      ["besan", "gram flour"],
    "oats":       ["oats"],
    "coffee":     ["coffee"],
    "tea":        ["tea", "chai"],
    "spice":      ["jeera", "cumin", "turmeric", "haldi", "chilli", "mirchi",
                   "garam masala", "coriander", "dhania", "mustard", "rai",
                   "black pepper", "pepper", "hing", "asafoetida"],
    "dry_fruit":  ["raisins", "kishmish", "almond", "badam", "cashew", "kaju", "pista",
                   "walnut", "akhrot", "dates", "khajoor"]
}

# Classes to *prefer* when using COCO fallback (not strict filter; we still OCR everything)
LIKELY_CONTAINER_COCO = {
    "bottle","cup","bowl","wine glass","vase","book","knife","spoon","fork","remote","cell phone","banana",
    "apple","orange","sandwich","broccoli","carrot","donut","cake","pizza","hot dog","chair","potted plant","bird"
}

OPEN_VOCAB_PROMPTS = [
    "jar","bottle","packet","pouch","box","tin","can","plastic container","steel container","spice jar",
    "snack packet","carton","bag","sachet"
]

# ----------------- Helpers -----------------
def build_lookup_syns():
    flat = []
    for canon, syns in PANTRY_KEYWORDS.items():
        for s in syns:
            flat.append((canon, s.lower()))
    return flat

LOOKUP = build_lookup_syns()

def best_keyword_match(text, cutoff=78):
    if not text:
        return None, 0
    txt = text.lower()
    candidates = [syn for _, syn in LOOKUP]
    m = process.extractOne(txt, candidates, scorer=fuzz.token_set_ratio, score_cutoff=cutoff)
    if not m:
        return None, 0
    matched_syn, score, _ = m
    for canon, syn in LOOKUP:
        if syn == matched_syn:
            return canon, score
    return None, 0

def try_load_yoloworld():
    for w in ["yolov8x-world.pt","yolov8l-world.pt","yolov8m-world.pt"]:
        if Path(w).exists():
            return YOLO(w)
    return None

def load_detector():
    m = try_load_yoloworld()
    if m:
        m.set_classes(OPEN_VOCAB_PROMPTS)  # open-vocab prompts
        return m, "world"
    return YOLO("yolov8x.pt"), "coco"      # downloads if missing

def color(i):
    rng = np.random.default_rng(i)
    return (int(rng.integers(40, 230)), int(rng.integers(40, 230)), int(rng.integers(40, 230)))

def cluster_shelves(img_h, y1, y2, bands=5):
    cy = (y1 + y2) / 2
    band_h = img_h / bands
    return int(np.clip(cy // band_h, 0, bands-1))

# ----------------- OCR -----------------
def ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

def ocr_text(reader, crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h,w = rgb.shape[:2]
    scale = 1.6
    rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    txts = reader.readtext(rgb, detail=0, paragraph=True)
    return " ".join(txts)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Path to image file")
    ap.add_argument("--video", help="Path to video file or webcam index")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--shelves", type=int, default=5, help="How many shelf bands to split vertically")
    args = ap.parse_args()

    if not args.image and args.video is None:
        ap.error("either --image or --video required")

    os.makedirs(args.outdir, exist_ok=True)

    model, mode = load_detector()
    rdr = ocr_reader()

    if args.image:
        crops_dir = os.path.join(args.outdir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(args.image)
        H,W = img.shape[:2]

        results = model.predict(source=args.image, imgsz=args.imgsz, conf=args.conf, verbose=False)

        annotated = img.copy()
        rows = []

        for r in results:
            if r.boxes is None:
                continue
            for i, b in enumerate(r.boxes):
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                label = r.names.get(cls_id, f"class_{cls_id}")

                # crop + OCR + keyword classify
                crop = img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                crop_id = f"ctr_{len(rows):03d}"
                crop_path = os.path.join(crops_dir, f"{crop_id}.jpg")
                cv2.imwrite(crop_path, crop)

                text = ocr_text(rdr, crop)
                item, score = best_keyword_match(text, cutoff=78)

                shelf_id = cluster_shelves(H, y1, y2, bands=args.shelves)

                # Draw
                disp = item if item else label
                c = color(len(rows))
                cv2.rectangle(annotated, (x1,y1), (x2,y2), c, 2)
                tag = f"{disp} {conf:.2f} | S{shelf_id}"
                cv2.rectangle(annotated, (x1, max(0,y1-22)), (x1+8*len(tag), y1), c, -1)
                cv2.putText(annotated, tag, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,20,20), 1, cv2.LINE_AA)

                rows.append({
                    "container_id": crop_id,
                    "detector_mode": mode,
                    "det_label": label,
                    "confidence": round(conf,3),
                    "shelf_id": shelf_id,
                    "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                    "ocr_text": text,
                    "item_type": item,
                    "item_conf": score,
                    "crop_path": crop_path
                })

        # Save outputs
        ann_path = os.path.join(args.outdir, "annotated.jpg")
        csv_path = os.path.join(args.outdir, "items.csv")
        cv2.imwrite(ann_path, annotated)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Quick summary (counts by item_type)
        if rows:
            df = pd.DataFrame(rows)
            summary = (df.assign(item_type=df["item_type"].fillna("unknown"))
                         .groupby(["shelf_id","item_type"]).size()
                         .reset_index(name="count")
                         .sort_values(["shelf_id","count"], ascending=[True, False]))
            summary.to_csv(os.path.join(args.outdir, "summary_by_shelf.csv"), index=False)

        print(f"[✓] saved: {ann_path}")
        print(f"[✓] saved: {csv_path}")
        if rows:
            print(f"[i] also wrote summary_by_shelf.csv and {len(rows)} crops")
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
            for r in model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False, stream=True):
                if r.boxes is None:
                    continue
                for i, b in enumerate(r.boxes):
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    label = r.names.get(cls_id, f"class_{cls_id}")
                    crop = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                    text = ocr_text(rdr, crop)
                    item, score = best_keyword_match(text, cutoff=78)
                    shelf_id = cluster_shelves(H, y1, y2, bands=args.shelves)
                    disp = item if item else label
                    c = color(i)
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), c, 2)
                    tag = f"{disp} {conf:.2f} | S{shelf_id}"
                    cv2.rectangle(annotated, (x1, max(0,y1-22)), (x1+8*len(tag), y1), c, -1)
                    cv2.putText(annotated, tag, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(20,20,20),1,cv2.LINE_AA)
                    summary[disp] = summary.get(disp,0) + 1
            cv2.imshow("pantry", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if summary:
            print("[summary] detections:")
            for k,v in summary.items():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
