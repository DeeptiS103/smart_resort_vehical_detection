# Interactive vehicle detection comparison dashboard for the local YOLO dataset.

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml
from ultralytics import YOLO


# Resolve dataset and model assets relative to the application directory.
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = (
    BASE_DIR
    / "archive"
    / "No_Apply_Grayscale"
    / "No_Apply_Grayscale"
    / "Vehicles_Detection.v8i.yolov8"
)
IMAGE_DIR = DATASET_DIR / "train" / "images"
LABEL_DIR = DATASET_DIR / "train" / "labels"
DATA_YAML = DATASET_DIR / "data.yaml"
MODEL_PATH = BASE_DIR / "yolov8n.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VEHICLE_CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


# Configure the dashboard for a widescreen analytical layout.
st.set_page_config(
    page_title="Smart Resort Vehicle Detection & Analytics",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply a restrained visual theme without changing Streamlit's interaction model.
st.markdown(
    """
    <style>
    :root {
        --ink: #172b27;
        --muted: #63736c;
        --forest: #174b3d;
        --mint: #dcefe4;
        --line: #dce4dd;
        --paper: #f4f7f2;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 92% 0%, rgba(181, 220, 197, .27), transparent 32%),
            var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #edf3ed;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }
    .dashboard-kicker {
        color: var(--forest);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: .25rem 0 .55rem;
    }
    h1, h2, h3 { color: var(--ink); }
    h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.35rem;
        line-height: 1.15;
        margin-bottom: .3rem;
    }
    .dashboard-subtitle { color: var(--muted); margin-bottom: 1.6rem; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, .86);
        border: 1px solid var(--line);
        border-top: 3px solid var(--forest);
        padding: 1rem 1.1rem;
        border-radius: 6px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load dataset metadata once so labels use the dataset's own class names.
@st.cache_data(show_spinner=False)
def load_class_names(metadata_path: str) -> list[str]:
    # Read dataset class names from the Roboflow-generated YAML metadata.
    with Path(metadata_path).open("r", encoding="utf-8") as metadata_file:
        metadata = yaml.safe_load(metadata_file) or {}
    names: Any = metadata.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


@st.cache_resource(show_spinner="Loading YOLOv8n weights...")
def load_model(model_path: str) -> YOLO:
    # Load and cache the pretrained YOLOv8n model for Streamlit reruns.
    return YOLO(model_path)


def list_sample_images(image_dir: Path) -> list[Path]:
    # Return supported dataset images in a stable, user-friendly order.
    return sorted(
        (path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


# Convert normalized YOLO annotations into clipped source-image pixel boxes.
def read_ground_truth(label_path: Path, image_width: int, image_height: int, class_names: list[str]) -> list[dict[str, Any]]:
    # Parse normalized YOLO labels and convert each box to pixel coordinates.
    if not label_path.is_file():
        return []

    boxes: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            st.warning(f"Skipping malformed label row {line_number} in {label_path.name}.")
            continue
        try:
            class_id = int(values[0])
            center_x, center_y, box_width, box_height = map(float, values[1:])
        except ValueError:
            st.warning(f"Skipping non-numeric label row {line_number} in {label_path.name}.")
            continue

        x_min = round((center_x - box_width / 2) * image_width)
        y_min = round((center_y - box_height / 2) * image_height)
        x_max = round((center_x + box_width / 2) * image_width)
        y_max = round((center_y + box_height / 2) * image_height)
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"Class {class_id}"
        boxes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "x_min": max(0, min(image_width, x_min)),
                "y_min": max(0, min(image_height, y_min)),
                "x_max": max(0, min(image_width, x_max)),
                "y_max": max(0, min(image_height, y_max)),
            }
        )
    return boxes


def draw_boxes(image_bgr: Any, boxes: list[dict[str, Any]], color: tuple[int, int, int], show_confidence: bool = False) -> Any:
    # Draw labeled bounding boxes on a copy of an OpenCV BGR image.
    annotated = image_bgr.copy()
    for index, box in enumerate(boxes, start=1):
        top_left = (box["x_min"], box["y_min"])
        bottom_right = (box["x_max"], box["y_max"])
        cv2.rectangle(annotated, top_left, bottom_right, color, 2)
        label = f"{box['class_name']} {box['confidence']:.2f}" if show_confidence else box["class_name"]
        text_y = max(18, top_left[1] - 7)
        cv2.putText(
            annotated,
            f"{index}. {label}",
            (top_left[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


# Run pretrained COCO inference and keep vehicle categories for comparison.
def predict_vehicles(model: YOLO, image_bgr: Any, confidence_threshold: float) -> list[dict[str, Any]]:
    # Run YOLOv8n and retain its COCO-labeled car, motorcycle, bus, and truck boxes.
    results = model.predict(source=image_bgr, conf=confidence_threshold, verbose=False)
    if not results or results[0].boxes is None:
        return []

    predictions: list[dict[str, Any]] = []
    result = results[0]
    for coordinates, confidence, class_value in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        class_id = int(class_value)
        if class_id not in VEHICLE_CLASS_NAMES:
            continue
        x_min, y_min, x_max, y_max = (int(round(value)) for value in coordinates)
        predictions.append(
            {
                "class_id": class_id,
                "class_name": VEHICLE_CLASS_NAMES[class_id],
                "confidence": float(confidence),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )
    return predictions


# Render both annotation sources with consistent source-image dimensions.
def render_comparison(image_bgr: Any, ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> plt.Figure:
    # Render ground truth and model predictions side by side with Matplotlib.
    ground_truth_image = draw_boxes(image_bgr, ground_truth, (255, 70, 45))
    prediction_image = draw_boxes(image_bgr, predictions, (65, 190, 125), show_confidence=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#f4f7f2")
    panels = [
        (axes[0], ground_truth_image, f"Ground Truth  /  {len(ground_truth)} objects"),
        (axes[1], prediction_image, f"YOLOv8n Predictions  /  {len(predictions)} objects"),
    ]
    for axis, annotated_image, title in panels:
        axis.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        axis.set_title(title, loc="left", color="#172b27", fontsize=12, fontweight="bold", pad=12)
        axis.axis("off")
    figure.tight_layout(pad=1.8)
    return figure


def build_prediction_table(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    # Format model outputs as a stable, report-ready table.
    columns = ["ID", "Class Name", "Confidence", "X Min (px)", "Y Min (px)", "X Max (px)", "Y Max (px)"]
    rows = [
        {
            "ID": index,
            "Class Name": item["class_name"],
            "Confidence": item["confidence"],
            "X Min (px)": item["x_min"],
            "Y Min (px)": item["y_min"],
            "X Max (px)": item["x_max"],
            "Y Max (px)": item["y_max"],
        }
        for index, item in enumerate(predictions, start=1)
    ]
    return pd.DataFrame(rows, columns=columns)


# Build sidebar controls and validate the expected local dataset structure.
st.sidebar.markdown("## Analysis controls")
st.sidebar.caption("Select a training image and adjust the detector confidence threshold.")
confidence_threshold = st.sidebar.slider(
    "YOLO confidence threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    format="%.2f",
)

if not IMAGE_DIR.is_dir():
    st.error(f"Dataset image directory was not found: {IMAGE_DIR}")
    st.stop()
if not DATA_YAML.is_file():
    st.error(f"Dataset metadata was not found: {DATA_YAML}")
    st.stop()

sample_images = list_sample_images(IMAGE_DIR)
if not sample_images:
    st.error(f"No supported images were found in {IMAGE_DIR}.")
    st.stop()

selected_image = st.sidebar.selectbox(
    "Sample image",
    sample_images,
    format_func=lambda path: path.name,
)

st.markdown('<p class="dashboard-kicker">Computer vision · object detection</p>', unsafe_allow_html=True)
st.title("Smart Resort Vehicle Detection & Analytics", anchor=False)
st.markdown(
    '<p class="dashboard-subtitle">Compare annotated training data with pretrained YOLOv8n vehicle detections.</p>',
    unsafe_allow_html=True,
)
st.caption(f"TRAIN SPLIT  /  {len(sample_images):,} images  /  {selected_image.name}")

image_bgr = cv2.imread(str(selected_image))
if image_bgr is None:
    st.error(f"OpenCV could not read {selected_image.name}.")
    st.stop()
image_height, image_width = image_bgr.shape[:2]

try:
    class_names = load_class_names(str(DATA_YAML))
    ground_truth = read_ground_truth(
        LABEL_DIR / f"{selected_image.stem}.txt",
        image_width,
        image_height,
        class_names,
    )
    with st.spinner("Running vehicle detection..."):
        model = load_model(str(MODEL_PATH))
        predictions = predict_vehicles(model, image_bgr, confidence_threshold)
except Exception as error:
    st.error(f"Unable to complete detection: {error}")
    st.stop()

label_path = LABEL_DIR / f"{selected_image.stem}.txt"
if not label_path.is_file():
    st.warning("No matching ground-truth label file was found for this image.")

# Summarize outputs, show the comparison, and expose report-ready detections.
average_confidence = (
    sum(item["confidence"] for item in predictions) / len(predictions) if predictions else None
)
metric_columns = st.columns(3)
metric_columns[0].metric("Ground-truth objects", f"{len(ground_truth):,}")
metric_columns[1].metric("Detected vehicles", f"{len(predictions):,}")
metric_columns[2].metric(
    "Average confidence",
    f"{average_confidence:.1%}" if average_confidence is not None else "N/A",
)

st.subheader("Visual comparison", anchor=False)
figure = render_comparison(image_bgr, ground_truth, predictions)
st.pyplot(figure, use_container_width=True)
plt.close(figure)

st.subheader("Detection records", anchor=False)
st.caption("Coordinates are pixel values in the source image; confidence is the model score.")
prediction_table = build_prediction_table(predictions)
st.dataframe(
    prediction_table,
    use_container_width=True,
    hide_index=True,
    column_config={"Confidence": st.column_config.NumberColumn(format="%.4f")},
)
st.download_button(
    "Download detections as CSV",
    data=prediction_table.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_image.stem}_detections.csv",
    mime="text/csv",
    icon=":material/download:",
)
