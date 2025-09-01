# Grocery-Inventory-Management

Utilities for detecting and labeling pantry items from images.

Both `detect_pantry.py` and `pantry_detect_and_label.py` write an
annotated image along with CSV summaries.  By default the scripts also
open a window showing the annotated image.  Window display requires a
GUI-capable OpenCV build such as **opencv-python**.  If you're running in
a headless environment or using `opencv-python-headless`, disable the
window with `--no-display`:

```bash
python detect_pantry.py --image path/to/pantry.jpg --no-display
python pantry_detect_and_label.py --image path/to/pantry.jpg --no-display
```

The `--display/--no-display` flag pair controls whether the window is
shown (it defaults to enabled).

The simple Flask web app in `webapp/` now supports capturing a short
video clip from your webcam.  Use the **Start Video Stream** button to
record and **Stop Video Stream** to finish; the clip's first frame is
sent through the same detection pipeline as an uploaded image.

