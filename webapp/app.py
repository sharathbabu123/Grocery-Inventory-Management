import os
import tempfile
import uuid
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory
import subprocess
import pandas as pd
import requests
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BRANDS = ["Brand A", "Brand B", "Brand C"]
QUANTITIES = list(range(1, 11))

def detect_items(image_path):
    """Run detect_pantry.py and return list of dicts with name and crop image."""
    tmpdir = tempfile.mkdtemp()
    script = Path(__file__).resolve().parent.parent / 'detect_pantry.py'
    items = []
    try:
        subprocess.run(
            ['python', str(script), '--image', str(image_path), '--outdir', tmpdir],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        csv_path = Path(tmpdir) / 'detections.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            img = cv2.imread(str(image_path))
            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                crop = img[y1:y2, x1:x2]
                fname = f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(app.config['UPLOAD_FOLDER'] / fname), crop)
                items.append({'name': str(row['label']), 'image': fname})
    except Exception as e:
        print(f'detection failed: {e}')
    return items

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('pantry_image')
        scanned_codes = request.form.get('qr_items', '')
        items = [{'name': c.strip(), 'image': None} for c in scanned_codes.split(',') if c.strip()]
        if file and file.filename:
            save_path = app.config['UPLOAD_FOLDER'] / file.filename
            file.save(save_path)
            items.extend(detect_items(save_path))
        return render_template('list.html', items=items, brands=BRANDS, quantities=QUANTITIES)
    return render_template('index.html')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/order', methods=['POST'])
def order():
    total = int(request.form.get('total', 0))
    selected = []
    for i in range(total):
        item = request.form.get(f'item-{i}')
        if item:
            brand = request.form.get(f'brand-{i}')
            qty = request.form.get(f'qty-{i}')
            selected.append({'item': item, 'brand': brand, 'quantity': qty})
    try:
        requests.post('https://httpbin.org/post', json=selected, timeout=5)
    except Exception as e:
        print(f'order request failed: {e}')
    return render_template('ordered.html', items=selected)

if __name__ == '__main__':
    app.run(debug=True)
