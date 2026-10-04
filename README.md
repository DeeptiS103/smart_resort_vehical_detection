# Smart Resort Vehicle Detection & Analytics

A Streamlit dashboard for comparing YOLO-format ground-truth annotations with vehicle detections from pretrained YOLOv8n.

## Run locally

Use Python 3.10 or newer, then from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On first launch, Ultralytics downloads `yolov8n.pt` if it is not already available locally. The training image, label, and `data.yaml` paths are resolved relative to this project directory.

## Analysis notes

- Ground-truth boxes are read from the matching YOLO `.txt` file and converted from normalized center/size coordinates to source-image pixels.
- Ground-truth boxes are drawn in blue. Model predictions are drawn in green.
- The pretrained COCO model reports car, motorcycle, bus, and truck detections. It has no separate pickup category, although `Pickup` is present in the dataset labels.
- Object totals and mean confidence are descriptive. The dashboard does not perform one-to-one IoU matching, so these values should not be interpreted as precision, recall, or accuracy.
- Model weights are pretrained COCO weights, not weights fine-tuned on this local dataset.