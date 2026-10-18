# Urban Street Vehicle Detection & InceptionV3 Analytics: Technical Report

## Executive Summary and Urban Traffic Context

Urban transport agencies and municipal operators require reliable observations of traffic volume and vehicle mix to understand network use and plan operations. The Federal Highway Administration, for example, identifies traffic volume and vehicle classification among core highway monitoring data (FHWA, n.d.). A vision system that detects vehicles in street-camera imagery could contribute to intersection and corridor counts, curbside occupancy analysis, bus or freight movement studies, and identification of recurring congestion patterns. These are potential downstream uses, not outputs currently computed by this project. A still-image vehicle box does not measure speed, establish a vehicle's route, identify a driver, or provide an enforcement-grade record; speed and flow analyses require calibrated geometry and/or temporal tracking, and enforcement requires separate validation, governance and legal authorization.

The project is a Streamlit image-analysis dashboard that applies an ImageNet-pretrained InceptionV3 classifier and Grad-CAM visual explanations to one selected image. It reads YOLO-format annotations, converts normalized box coordinates to source-image pixels, and displays the annotations alongside regions heuristically extracted from an InceptionV3 activation map. The active project title is “Urban Street Vehicle Detection & InceptionV3 Analytics” ([app.py](app.py#L404)). The README retains a legacy title that does not match the active application; that older branding is not used to define this report's scope. Dataset metadata identifies a private Roboflow vehicle-detection project, version 8, with Bus, Car, Motorcycle, Pickup and Truck classes ([active data.yaml](archive/No_Apply_Grayscale/No_Apply_Grayscale/Vehicles_Detection.v8i.yolov8/data.yaml)). Although the wider assignment describes a Kaggle vehicle dataset, the local files do not independently verify Kaggle provenance.

The report has three objectives: document the annotation-to-visualization pipeline, critically distinguish classifier-derived Grad-CAM regions from native object detections, and assess what additional evaluation and governance are needed before urban traffic use. The scope is limited to offline image analysis; it does not evaluate live video, tracking, traffic control or enforcement.

The dashboard demonstrates the distinction between image classification explanations and object detection, but it is not a dual-model inference system. No YOLO inference call occurs in [app.py](app.py), and the dependency list does not include Ultralytics. A `yolov8n.pt` checkpoint exists in the project root but is not loaded by the application. Accordingly, the report compares the implemented InceptionV3 localization heuristic with the theoretical capabilities of a native detector; it does not claim an empirical YOLOv8-versus-InceptionV3 benchmark.

The active dataset contains 143 images: 102 training, 28 validation and 13 test images. Their label files contain 2,069, 563 and 254 bounding-box rows respectively. The dashboard enumerates only the training split. No detector training routine, YOLO validation run, IoU-based matching procedure, latency logger, or false-positive/false-negative calculation is present. Dataset counts are therefore reported as descriptive evidence, while detection quality, operational confidence distributions and latency remain unmeasured. This distinction precludes production-readiness claims for urban traffic monitoring.

## Literature Review and Theoretical Framework

### Image classification and object detection

Image classification assigns one or more semantic categories to an image as a whole. Inception architectures improve computational efficiency through parallel convolutional branches and factorized operations; InceptionV3 is a classification network commonly used with ImageNet weights (Szegedy et al., 2016; TensorFlow, n.d.-a). In this project, the Keras model retains its ImageNet classification head and accepts an image resized to 299 × 299 pixels. It returns class scores over 1,000 ImageNet categories, rather than a set of spatially distinct vehicle instances ([model construction and inference](app.py#L116), [Grad-CAM pipeline](app.py#L217)).

Object detection instead estimates a category, confidence and spatial extent for each object instance. The original YOLO formulation cast detection as a single network prediction over a grid, emphasizing joint localization and classification (Redmon et al., 2016). SSD predicts detections at multiple feature-map scales and is a representative dense detector (Liu et al., 2016). These detector families naturally return multiple candidate boxes in one image, an important property for busy intersections, multilane roads and congested curb areas (Sivaraman and Trivedi, 2013). InceptionV3 can provide image-level evidence or an explainability demonstration, but its classification output does not by itself satisfy per-vehicle counting or localization requirements.

The project contains no executable SSD path. Ultralytics documents YOLOv8 as an object-detection family with an anchor-free split head and provides a Python interface for model loading, prediction, validation and training (Ultralytics, n.d.-a; Ultralytics, n.d.-b). The presence of a checkpoint alone does not mean that this capability has been wired into this application. In this repository, the practical comparison is therefore between one classifier-derived region heuristic and human-authored dataset boxes, not between two running models.

### Anchors, box regression and suppression

In anchor-based detectors, a collection of predefined reference rectangles, or anchors, is placed at feature-map locations and scales. A network predicts class scores and offsets that transform a selected anchor into a candidate box. A common parameterization is

$$
x = x_a + t_x w_a,\qquad y = y_a + t_y h_a,\qquad
w = w_a e^{t_w},\qquad h = h_a e^{t_h},
$$

where $(x_a,y_a,w_a,h_a)$ is an anchor and $(t_x,t_y,t_w,t_h)$ are learned offsets. Bounding-box regression trains these offsets to reduce a localization loss, usually in conjunction with classification and objectness-related losses. Anchors are a useful conceptual framework for SSD and earlier detector designs, but should not be attributed indiscriminately to YOLOv8: Ultralytics describes the YOLOv8 head as anchor-free (Ultralytics, n.d.-a). Anchor-free designs predict locations and box dimensions without matching each location to a set of preset anchor shapes.

Dense models produce overlapping candidate boxes. Non-Maximum Suppression (NMS) orders candidates by score and removes lower-scoring boxes whose intersection over union (IoU) with a higher-scoring box exceeds a chosen threshold. For boxes $A$ and $B$, $\operatorname{IoU}(A,B)=|A\cap B|/|A\cup B|$. NMS reduces duplicate detections but can suppress a genuine vehicle when two nearby instances overlap strongly; class-aware versus class-agnostic filtering and threshold selection affect this trade-off. NMS is a post-processing procedure, not a substitute for a detector learning accurate instance boundaries (Hosang, Benenson and Schiele, 2017).

### Grad-CAM and heuristic region extraction

Grad-CAM produces a class-conditioned spatial relevance map from a convolutional feature tensor. Let $A^k_{ij}$ denote activation channel $k$ at spatial position $(i,j)$, and let $y^c$ be the score for target class $c$. The channel importance weights and positive class-activation map are

$$
\alpha_k^c = \frac{1}{Z}\sum_i\sum_j\frac{\partial y^c}{\partial A^k_{ij}},\qquad
L^c_{\mathrm{Grad\text{-}CAM}} = \operatorname{ReLU}\left(\sum_k \alpha_k^c A^k\right).
$$

This map indicates image regions that contribute positively to a selected class score; it is not a segmentation mask, an objectness map or a detector output (Selvaraju et al., 2017). In the application, gradients are taken with respect to InceptionV3's `mixed10` tensor. The resulting map is normalized, resized to source-image dimensions, thresholded, opened morphologically, and converted from external contours to axis-aligned rectangles ([implementation](app.py#L217), [region extraction](app.py#L265)).

For a heatmap $H$, threshold $\tau$ yields a binary mask $B(p)=1[H(p)\geq\tau]$. Morphological opening by structuring element $K$ is erosion followed by dilation, $B\circ K=(B\ominus K)\oplus K$; it tends to remove small foreground speckles but also erodes thin or small regions (OpenCV, n.d.). The app then discards contours smaller than a selected image-area fraction and encloses each remaining contour in a rectangle. Thus, Grad-CAM plus morphology adapts a classifier's coarse spatial explanation into candidate regions, but it does not train the classifier to localize objects. Threshold, kernel and area filtering introduce tunable heuristic behavior; they cannot recover instance boundaries absent from the activation map.

The distinction between scores is also material. The app's “Vehicle class score” is the selected ImageNet whole-image class score. “Mean Activation” is the average normalized heatmap value inside a candidate contour. Neither is a per-box detector confidence, and neither is calibrated by the application as the probability that a particular rectangle contains a vehicle.

## System Architecture and Methodology

### Dataset and coordinate pipeline

The active application paths resolve relative to the project root and target `archive/No_Apply_Grayscale/No_Apply_Grayscale/Vehicles_Detection.v8i.yolov8`. `app.py` points specifically to `train/images`, `train/labels` and the split's `data.yaml` ([path configuration](app.py#L17)). The YAML declares five classes in this order: Bus (0), Car (1), Motorcycle (2), Pickup (3) and Truck (4). Other grayscale, YOLOv9, COCO and TensorFlow exports exist in the archive, but are not read by the active dashboard.

Each YOLO detection annotation is represented as `class_id x_center y_center width height`, with center and dimensions normalized by image width and height. Given source image width $W$ and height $H$, the app maps the normalized center-size representation to corners:

$$
x_{\min}=\operatorname{round}\left((x_c-w/2)W\right),\quad
y_{\min}=\operatorname{round}\left((y_c-h/2)H\right),\\
x_{\max}=\operatorname{round}\left((x_c+w/2)W\right),\quad
y_{\max}=\operatorname{round}\left((y_c+h/2)H\right).
$$

The results are clipped to the source-image bounds. The parser checks for five fields, converts numeric values, resolves class IDs against YAML names and warns on malformed or non-numeric rows ([annotation conversion](app.py#L147)). An audit that treats each label file independently found 2,069, 563 and 254 five-field annotation rows in train, validation and test, with no nonstandard non-empty rows. The source dimensions are retained for drawing; coordinates shown in the dashboard are pixels in the original image, not the 299 × 299 classifier input frame.

### Dashboard and inference flow

The Streamlit interface loads class names from YAML, enumerates supported training images, lets an operator select an image, and presents ground-truth annotations beside Grad-CAM-derived regions. InceptionV3 is created with ImageNet weights and cached as a resource; a second Keras model exposes both `mixed10` and the classifier output. The ImageNet synset-to-index mapping is cached separately ([model setup](app.py#L105)).

For a selected image, OpenCV reads a BGR image; the inference path converts it to RGB, resizes it to 299 × 299, applies TensorFlow's InceptionV3 preprocessing and creates a batch dimension. TensorFlow documents this preprocessing as scaling pixel values to the range $[-1,1]$ (TensorFlow, n.d.-b). The application decodes the ImageNet predictions, searches the top 1,000 decoded categories for a vehicle-related label, and selects the highest-scoring matching category. If no matching label is found, it instead explains the top non-vehicle category and displays a warning. This fallback means a heatmap can be shown even when the classifier has not recognized a vehicle.

Gradients of the selected class score with respect to `mixed10` are spatially averaged per channel. The weighted feature maps are summed, passed through ReLU and normalized by their maximum. The heatmap is resized to the source dimensions. A threshold slider (default 0.80), elliptical kernel slider (default 1 × 1), opening-iteration slider (default 1) and minimum-area slider (default 0.10% of the image) control extraction. With a 1 × 1 kernel, morphological opening is effectively an identity operation, so the default configuration does not remove speckle through kernel shape. The class score, candidate-region count and ground-truth count are displayed, while region coordinates and activation statistics can be exported to CSV ([controls and output](app.py#L351), [dashboard results](app.py#L448)).

The user interface has two visual comparison panels, but both do not represent two inference engines: one contains dataset annotations and the other contains InceptionV3-derived regions. No `ultralytics` import, `YOLO(...)` model initialization or YOLO prediction call occurs in the application, and `requirements.txt` lists Streamlit, TensorFlow, OpenCV, Matplotlib, pandas and PyYAML, but not Ultralytics. The root checkpoint is consequently an unused asset from the perspective of the current executable path.

### Algorithmic comparison

| Aspect | Implemented InceptionV3 + Grad-CAM | Native YOLOv8 detector (not implemented in this dashboard) |
|---|---|---|
| Learned task | ImageNet whole-image classification | Per-instance object detection |
| Spatial output | Thresholded explanation contours converted to rectangles | Candidate boxes with class labels and detection scores |
| Vehicle taxonomy | ImageNet labels searched by a hand-maintained term list | Dataset-specific classes require suitable trained weights |
| Multiple vehicles | May highlight one dominant vehicle, merge activations or fragment one vehicle | Can return multiple instances; overlap and occlusion still affect recall and NMS |
| Confidence meaning | Whole-image class score; not a box confidence | Per-detection score, subject to model and post-processing configuration |
| Training in repository | No local fine-tuning; ImageNet pretrained weights | No training or inference code; local checkpoint is not loaded |
| Appropriate use here | Explainability demonstration and rough visual region proposal | Candidate operational detector after integration and held-out evaluation |

For a single, isolated vehicle, a class activation map may produce a plausible region around salient vehicle features, but its rectangle can be loose and may include road or background context. In dense urban traffic, vehicles may overlap in the image, queue at signals, or appear small in distant lanes. The global class target has no explicit mechanism for separating instances: connected activation can merge adjacent vehicles, while weakly activated parts can fragment one vehicle. These are architecture-based expectations, not measured findings from a completed qualitative evaluation. A trained detector is structurally better suited to output separate boxes, although occlusion, motion blur, small apparent size, reflections, nighttime lighting and NMS settings can still cause missed or duplicate detections. Actual localization quality requires held-out predictions matched to reference boxes.

## Empirical Results and Comparative Analysis

### Reproducible project inventory

| Split | Images | Label files | Ground-truth boxes |
|---|---:|---:|---:|
| Train | 102 | 102 | 2,069 |
| Validation | 28 | 28 | 563 |
| Test | 13 | 13 | 254 |
| **Total** | **143** | **143** | **2,886** |

The box-level class totals across the packaged dataset are Bus 35, Car 2,104, Motorcycle 411, Pickup 278 and Truck 58. Cars account for approximately 72.9% of annotated boxes, motorcycles 14.2%, pickups 9.6%, trucks 2.0% and buses 1.2%. This class imbalance, and the small 13-image test split, constrain any generalization claim. The dashboard itself displays the training split only; it does not expose a test-set evaluation workflow.

### Metrics and evidence limits

| Requested measure | Evidence in current project | Reported result |
|---|---|---|
| InceptionV3 localization precision / recall | No IoU matching against labels; no aggregate evaluation | Not measured |
| YOLOv8 precision / recall / mAP | No YOLO inference or validation call; checkpoint not loaded | Not measured |
| Confidence-score distribution | Selected image's whole-image ImageNet score is shown; no per-box scores or distribution logging | Not measured as a detection metric |
| Inference latency | No timing instrumentation, hardware record or controlled benchmark | Not measured |
| False positives / false negatives | No prediction-to-ground-truth assignment or confusion accounting | Not measured |
| Qualitative box precision | Dashboard can display one selected training image, but no documented annotated-case study is included | No empirical conclusion claimed |

The absence of these measures is not equivalent to zero errors or poor performance; it means the repository does not provide the observations needed to calculate them. Counts of ground-truth objects and Grad-CAM regions cannot be compared as detection accuracy because region count is not a one-to-one instance match. Similarly, an ImageNet classification score must not be presented as a YOLO-style confidence score.

A defensible next evaluation would freeze model weights and preprocessing, run both systems on the untouched test split and match predictions to ground truth class-wise with a predeclared IoU threshold. Each ground-truth box should be matched to at most one prediction. At an operating confidence threshold, precision is $TP/(TP+FP)$, recall is $TP/(TP+FN)$ and false-negative rate is $FN/(TP+FN)$. Average precision should be computed over a confidence sweep, with mAP reported at a clearly stated IoU criterion or range. Results should be stratified by class, object size, crowding/overlap and lighting, and confidence intervals should reflect the limited number of test images. Latency should be recorded after warm-up on a named device and software stack, separating preprocessing, model inference and post-processing; report batch-one median and 95th-percentile timings. The same input resolution and measurement protocol should be used for each executable model.

## Strategic Recommendations and Operational Deployment

### Urban traffic operations and human oversight

If developed into a validated detector, the system could contribute to vehicle counts and class distributions at intersections, corridor-level traffic studies, curbside loading-zone occupancy, freight and bus-lane monitoring, and identification of recurring queue patterns. These applications require camera placement and sampling periods suited to the question being studied. Counts from a single view cannot by themselves reveal origin-destination paths, legal compliance or vehicle speed. Speed estimates require calibrated image geometry and temporal observations; movement patterns require tracking and appropriate association across frames. The present application processes selected still images and implements none of these functions.

Vehicle boxes should be treated as measurements for analyst or traffic-operator review, not as enforcement determinations. A detection does not establish right-of-way violation, parking legality, driver identity or intent. If the system informs signal timing or enforcement, a separate safety case, target-population validation, transparent operating thresholds, human review and applicable public-sector approval are required. The project's five classes omit pedestrians, cyclists, traffic signals and road geometry, so it cannot alone support a complete multimodal street-safety analysis.

### Deployment risks, privacy and scalability

Urban camera imagery varies with mounting height, lens, road width, camera angle, shadows from buildings, headlight glare, rain, fog, seasonal foliage, nighttime exposure and motion blur. Occlusion is common at multilane junctions and in queues. A model evaluated on the archived sample should not be assumed to generalize across neighborhoods, camera vendors or weather conditions. A deployment dataset should cover the intended streets and operating periods, record relevant scene conditions, and remain separate from training data for final evaluation.

Edge inference can reduce video backhaul and preserve service during network outages, but introduces constraints on accelerator availability, power, thermal limits, model size, maintenance and secure software updates. Centralized processing can simplify fleet management while increasing bandwidth, availability and data-governance requirements. Benchmark the chosen device with realistic camera resolution and frame rate, measuring acquisition, preprocessing, inference, post-processing, transmission and alert latency separately. Define health checks, failure behavior, retention of queued events and a manual operating alternative.

Street-camera footage can capture pedestrians, vehicle registration plates, travel patterns and time-location data. Before collection or deployment, the responsible authority should document purpose, legal basis, proportionality, access controls, retention periods, signage, encryption, auditability, data-sharing limits and incident response. Data minimization may include processing locally, discarding frames after aggregate counts are produced, masking nonessential regions, and excluding plate recognition unless independently justified. The GDPR applies where its territorial and material scope is met (European Parliament and Council, 2016); other jurisdictions impose their own surveillance and public-record requirements. Privacy and legal review should precede operational deployment.

### Delivery plan and business case

1. **Complete detector integration.** Add a supported detector inference path, verify checkpoint provenance and class mapping, and expose thresholds and input size clearly. Retain Grad-CAM as an explanatory comparison rather than treating it as instance detection.
2. **Build a representative benchmark.** Confirm dataset rights and provenance, check for train-test leakage, and collect additional permitted street scenes covering camera positions, traffic density, road users, weather and day/night variation. Keep a genuinely held-out test set.
3. **Evaluate operational performance.** Report per-class AP, precision, recall and false-negative rate at specified IoU and confidence thresholds. Add counts per camera-hour, missed-vehicle rate, p50/p95 latency, uptime, and separate results by vehicle size, congestion and illumination. Include analyst review of overlapping, small and partially occluded vehicles.
4. **Pilot in shadow mode.** Log proposed counts or alerts without controlling signals or initiating enforcement. Compare sampled outputs with manual counts and existing traffic data; quantify error and operational usefulness before integrating with traffic-management workflows.
5. **Assess public value and cost.** Estimate benefits from improved traffic-data coverage, reduced manual counting effort or more timely identification of recurring bottlenecks. Deduct camera, edge-compute, communications, integration, maintenance, security, privacy-governance and staffing costs. Calculate ROI as $(\text{annual net benefit}/\text{initial investment})\times100\%$ only from measured agency-specific inputs. The repository contains no cost or outcome data, so a numerical ROI claim is not supportable.

The current dashboard is best characterized as an educational and exploratory image-analysis prototype. It demonstrates YOLO annotation conversion and classifier-derived saliency regions, but does not yet provide a deployed vehicle detector or validated traffic-monitoring service. Claims about intersection counts, congestion response, safety outcomes or operational return require detector integration, representative test data, temporal processing where needed, governance approval and field evaluation.

## References

European Parliament and Council (2016) ‘Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 (General Data Protection Regulation)’, *Official Journal of the European Union*, L 119, pp. 1–88. Available at: https://eur-lex.europa.eu/eli/reg/2016/679/oj (Accessed: 10 October 2026).

Federal Highway Administration (FHWA) (n.d.) ‘Traffic Monitoring Analysis System (TMAS)’. Available at: https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/ (Accessed: 10 October 2026).

Hosang, J., Benenson, R. and Schiele, B. (2017) ‘Learning non-maximum suppression’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(2), pp. 201–210. doi: 10.1109/TPAMI.2016.2577036.

Liu, W. et al. (2016) ‘SSD: Single shot multibox detector’, in *Computer Vision – ECCV 2016*. Cham: Springer, pp. 21–37. doi: 10.1007/978-3-319-46448-0_2.

OpenCV (n.d.) ‘Morphological transformations’. Available at: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html (Accessed: 10 October 2026).

Redmon, J., Divvala, S., Girshick, R. and Farhadi, A. (2016) ‘You only look once: Unified, real-time object detection’, in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 779–788. doi: 10.1109/CVPR.2016.91.

Selvaraju, R.R. et al. (2017) ‘Grad-CAM: Visual explanations from deep networks via gradient-based localization’, in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 618–626. doi: 10.1109/ICCV.2017.74.

Sivaraman, S. and Trivedi, M.M. (2013) ‘Looking at vehicles on the road: A survey of vision-based vehicle detection, tracking, and behavior analysis’, *IEEE Transactions on Intelligent Transportation Systems*, 14(4), pp. 1773–1795. doi: 10.1109/TITS.2013.2266661.

Szegedy, C. et al. (2016) ‘Rethinking the Inception architecture for computer vision’, in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818–2826. doi: 10.1109/CVPR.2016.308.

Szeliski, R. (2022) *Computer Vision: Algorithms and Applications*. 2nd edn. Cham: Springer. doi: 10.1007/978-3-030-34372-9.

TensorFlow (n.d.-a) ‘tf.keras.applications.InceptionV3’. Available at: https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3 (Accessed: 10 October 2026).

TensorFlow (n.d.-b) ‘tf.keras.applications.inception_v3.preprocess_input’. Available at: https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input (Accessed: 10 October 2026).

Ultralytics (n.d.-a) ‘YOLOv8’. Available at: https://docs.ultralytics.com/models/yolov8/ (Accessed: 10 October 2026).

Ultralytics (n.d.-b) ‘Predict’. Available at: https://docs.ultralytics.com/modes/predict/ (Accessed: 10 October 2026).