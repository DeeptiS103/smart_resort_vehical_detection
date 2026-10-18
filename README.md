# Urban Street Vehicle Detection & InceptionV3 Analytics

A Streamlit dashboard that compares YOLO-format dataset annotations with approximate vehicle regions localized from pretrained ImageNet InceptionV3 using Grad-CAM and OpenCV contours.

## Run locally

Use Python 3.10-3.12 for broad TensorFlow wheel compatibility, then from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On first launch, Keras downloads the InceptionV3 ImageNet weights if they are not already cached. The training image, label, and `data.yaml` paths are resolved relative to this project directory.

## Analysis notes

- Ground-truth boxes are read from the matching YOLO `.txt` file and converted from normalized center/size coordinates to source-image pixels.
- Ground-truth boxes are drawn in blue; Grad-CAM contour regions use orange.
- InceptionV3 predicts an ImageNet class for the complete image. The app selects its highest-scoring vehicle category, differentiates that class score with respect to `mixed10`, and uses the resulting Grad-CAM heatmap as a spatial relevance estimate.
- OpenCV thresholds the heatmap, applies configurable morphological opening, filters small contours, and reports each remaining contour's pixel coordinates, area, class score, and mean activation.
- Grad-CAM regions are approximate explanatory regions, not learned object proposals. One region need not correspond to one vehicle; region counts are not detection accuracy, precision, or recall, and should not be evaluated as such without IoU-based matching.
- The classifier uses ImageNet weights and has not been fine-tuned on this local vehicle dataset.