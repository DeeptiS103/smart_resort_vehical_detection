# Interactive vehicle detection comparison dashboard for the local YOLO dataset.

from pathlib import Path
from typing import Any
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yaml


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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INCEPTION_INPUT_SIZE = (299, 299)
VEHICLE_LABEL_TERMS = {
    "ambulance", "bus", "cab", "car", "convertible", "fire engine", "jeep",
    "limousine", "minibus", "moped", "motor scooter", "motorcycle", "pickup",
    "racer", "recreational vehicle", "taxi", "tow truck", "trailer truck", "truck",
    "trolleybus", "van",
}


# Configure the dashboard for a widescreen analytical layout.
st.set_page_config(
    page_title="Urban Street Vehicle Detection & InceptionV3 Analytics",
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


@st.cache_resource(show_spinner="Loading InceptionV3 ImageNet weights...")
def load_inception_models() -> tuple[tf.keras.Model, tf.keras.Model]:
    # Cache ImageNet InceptionV3 and its mixed10 feature/classification bridge.
    classifier = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet")
    gradcam_model = tf.keras.Model(
        inputs=classifier.inputs,
        outputs=[classifier.get_layer("mixed10").output, classifier.output],
    )
    return classifier, gradcam_model


@st.cache_data(show_spinner=False)
def load_imagenet_synset_indices() -> dict[str, int]:
    # Map ImageNet synset strings returned by decode_predictions to logit indices.
    class_index_path = tf.keras.utils.get_file(
        "imagenet_class_index.json",
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
    )
    with Path(class_index_path).open("r", encoding="utf-8") as index_file:
        class_index_data = json.load(index_file)
    return {details[0]: int(index) for index, details in class_index_data.items()}


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


def draw_boxes(image_bgr: Any, boxes: list[dict[str, Any]], color: tuple[int, int, int], show_activation: bool = False) -> Any:
    # Draw ground-truth or Grad-CAM region boxes on a source-image copy.
    annotated = image_bgr.copy()
    for index, box in enumerate(boxes, start=1):
        top_left = (box["x_min"], box["y_min"])
        bottom_right = (box["x_max"], box["y_max"])
        cv2.rectangle(annotated, top_left, bottom_right, color, 2)
        label = f"R{index} {box['class_name']}"
        if show_activation:
            label = f"{label} {box['mean_activation']:.2f}"
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


def is_vehicle_label(label: str) -> bool:
    # Identify ImageNet labels representing whole vehicles, not vehicle parts.
    normalized = label.lower().replace("_", " ")
    if any(part in normalized for part in ("wheel", "tire", "seat", "mirror", "windshield")):
        return False
    return any(term in normalized for term in VEHICLE_LABEL_TERMS)


def compute_gradcam(gradcam_model: tf.keras.Model, image_bgr: Any) -> tuple[np.ndarray, str, float]:
    """Compute a class-specific Grad-CAM from InceptionV3's final mixed10 layer.
    InceptionV3 predicts an ImageNet class for the whole image rather than
    producing detections. The gradient of the strongest vehicle-class score
    with respect to mixed10 is pooled into channel weights; the weighted feature
    maps form a spatial relevance heatmap used downstream for approximate boxes.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, INCEPTION_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    input_tensor = tf.keras.applications.inception_v3.preprocess_input(
        np.expand_dims(resized.astype(np.float32), axis=0)
    )
    synset_indices = load_imagenet_synset_indices()

    with tf.GradientTape() as tape:
        convolution_features, class_scores = gradcam_model([input_tensor], training=False)
        decoded_classes = tf.keras.applications.inception_v3.decode_predictions(
            class_scores.numpy(), top=1000
        )[0]
        vehicle_classes = [item for item in decoded_classes if is_vehicle_label(item[1])]
        if vehicle_classes:
            selected_class = max(vehicle_classes, key=lambda item: item[2])
            class_name = selected_class[1].replace("_", " ").title()
            class_index = synset_indices[selected_class[0]]
        else:
            selected_class = decoded_classes[0]
            class_name = f"No vehicle class; top: {selected_class[1].replace('_', ' ').title()}"
            class_index = synset_indices[selected_class[0]]
        target_score = class_scores[:, class_index]

    gradients = tape.gradient(target_score, convolution_features)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients were not connected to the mixed10 feature map.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    class_activation = tf.reduce_sum(channel_weights * convolution_features, axis=-1)[0]
    class_activation = tf.maximum(class_activation, 0)
    maximum_activation = tf.reduce_max(class_activation)
    class_activation = tf.math.divide_no_nan(class_activation, maximum_activation)
    heatmap_small = class_activation.numpy().astype(np.float32)
    heatmap = cv2.resize(
        heatmap_small,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(heatmap, 0.0, 1.0), class_name, float(selected_class[2])


def localize_heatmap(heatmap: np.ndarray, threshold: float, kernel_size: int, opening_iterations: int, minimum_area_fraction: float, class_name: str, class_score: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    # Threshold Grad-CAM, open away speckle, then convert contours to regions.
    heatmap_u8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    _, binary_mask = cv2.threshold(
        heatmap_u8, int(round(threshold * 255)), 255, cv2.THRESH_BINARY
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=opening_iterations,
    )
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = heatmap.shape[0] * heatmap.shape[1] * minimum_area_fraction
    regions: list[dict[str, Any]] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue
        x_min, y_min, width, height = cv2.boundingRect(contour)
        contour_mask = np.zeros(heatmap.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        activation_values = heatmap[contour_mask > 0]
        regions.append(
            {
                "region_id": len(regions) + 1,
                "class_name": class_name,
                "class_score": class_score,
                "mean_activation": float(activation_values.mean()) if activation_values.size else 0.0,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_min + width,
                "y_max": y_min + height,
                "area_px": round(area),
                "area_percent": area / (heatmap.shape[0] * heatmap.shape[1]) * 100,
            }
        )
    return regions, opened_mask


# Render both annotation sources with consistent source-image dimensions.
def render_panel(
    image_bgr: Any,
    boxes: list[dict[str, Any]],
    title: str,
    color: tuple[int, int, int],
    show_activation: bool = False,
) -> plt.Figure:
    # Create one report-style Matplotlib panel with annotated source imagery.
    annotated = draw_boxes(image_bgr, boxes, color, show_activation=show_activation)
    figure, axis = plt.subplots(figsize=(7, 5.5), facecolor="#f4f7f2")
    axis.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axis.set_title(title, loc="left", color="#172b27", fontsize=12, fontweight="bold", pad=12)
    axis.axis("off")
    figure.tight_layout(pad=1.2)
    return figure


def build_prediction_table(regions: list[dict[str, Any]]) -> pd.DataFrame:
    # Format Grad-CAM contours as an exportable localization analysis table.
    columns = [
        "Region ID", "Vehicle Class", "Class Score", "Mean Activation",
        "X Min (px)", "Y Min (px)", "X Max (px)", "Y Max (px)",
        "Area (px)", "Image Area (%)",
    ]
    rows = [
        {
            "Region ID": item["region_id"],
            "Vehicle Class": item["class_name"],
            "Class Score": item["class_score"],
            "Mean Activation": item["mean_activation"],
            "X Min (px)": item["x_min"],
            "Y Min (px)": item["y_min"],
            "X Max (px)": item["x_max"],
            "Y Max (px)": item["y_max"],
            "Area (px)": item["area_px"],
            "Image Area (%)": item["area_percent"],
        }
        for item in regions
    ]
    return pd.DataFrame(rows, columns=columns)


# Expose localization and morphology controls for reproducible region extraction.
st.sidebar.markdown("## Localization controls")
st.sidebar.caption("Tune the class activation map and the contour-cleanup operation.")
localization_threshold = st.sidebar.slider(
    "Grad-CAM threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.80,
    step=0.05,
    format="%.2f",
)
morphology_kernel = st.sidebar.slider(
    "Morphological kernel size",
    min_value=1,
    max_value=21,
    value=1,
    step=2,
    help="Odd-sized elliptical kernel used by morphological opening.",
)
opening_iterations = st.sidebar.slider(
    "Opening iterations",
    min_value=1,
    max_value=5,
    value=1,
    step=1,
)
minimum_area_percent = st.sidebar.slider(
    "Minimum region area (% of image)",
    min_value=0.01,
    max_value=2.00,
    value=0.10,
    step=0.01,
    format="%.2f%%",
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

# st.markdown('<p class="dashboard-kicker">Computer vision · class activation localization</p>', unsafe_allow_html=True)
st.title("Urban Street Vehicle Detection & InceptionV3 Analytics", anchor=False)
st.markdown(
    '<p class="dashboard-subtitle">Explore class-specific InceptionV3 attention and compare its localized regions with dataset annotations.</p>',
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
    with st.spinner("Computing InceptionV3 Grad-CAM and vehicle regions..."):
        _, gradcam_model = load_inception_models()
        heatmap, vehicle_class, class_score = compute_gradcam(gradcam_model, image_bgr)
        regions, morphology_mask = localize_heatmap(
            heatmap,
            localization_threshold,
            morphology_kernel,
            opening_iterations,
            minimum_area_percent / 100.0,
            vehicle_class,
            class_score,
        )
except Exception as error:
    st.error(f"Unable to complete InceptionV3 localization: {error}")
    st.stop()

label_path = LABEL_DIR / f"{selected_image.stem}.txt"
if not label_path.is_file():
    st.warning("No matching ground-truth label file was found for this image.")
if vehicle_class.startswith("No vehicle class"):
    st.warning("ImageNet did not produce a vehicle class; the displayed heatmap explains the top non-vehicle class.")

# Summarize annotations, approximate regions, and the selected ImageNet category.
metric_columns = st.columns(3)
metric_columns[0].metric("Ground-truth objects", f"{len(ground_truth):,}")
metric_columns[1].metric("Candidate regions", f"{len(regions):,}")
metric_columns[2].metric("Vehicle class score", f"{class_score:.1%}", help=vehicle_class)

st.subheader("Ground truth vs. InceptionV3 localization", anchor=False)
ground_truth_column, localization_column = st.columns(2, gap="large")
with ground_truth_column:
    ground_truth_figure = render_panel(
        image_bgr,
        ground_truth,
        f"Dataset labels  /  {len(ground_truth)} objects",
        (255, 0, 0),
    )
    st.pyplot(ground_truth_figure, use_container_width=True)
    plt.close(ground_truth_figure)
with localization_column:
    localization_figure = render_panel(
        image_bgr,
        regions,
        f"Grad-CAM candidate regions  /  {len(regions)}",
        (0, 95, 255),
        show_activation=True,
    )
    st.pyplot(localization_figure, use_container_width=True)
    plt.close(localization_figure)

with st.expander("Inspect Grad-CAM heatmap and morphology mask"):
    heatmap_column, mask_column = st.columns(2)
    heatmap_column.image(heatmap, caption=f"mixed10 class activation · {vehicle_class}", width=True, clamp=True)
    mask_column.image(morphology_mask, caption="Thresholded and morphologically opened mask", width=True)

st.subheader("Localized region records", anchor=False)
st.caption("Coordinates and contour areas are source-image pixels; class score and mean activation are distinct quantities.")
prediction_table = build_prediction_table(regions)
st.dataframe(
    prediction_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Class Score": st.column_config.NumberColumn(format="%.4f"),
        "Mean Activation": st.column_config.NumberColumn(format="%.4f"),
        "Image Area (%)": st.column_config.NumberColumn(format="%.3f"),
    },
)
st.download_button(
    "Download region analysis as CSV",
    data=prediction_table.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_image.stem}_inceptionv3_regions.csv",
    mime="text/csv",
    icon=":material/download:",
)


