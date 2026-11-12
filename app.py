import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="FireGuard AI | Fire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Prefer the trained checkpoint. Set MODEL_PATH if your weights are elsewhere.
_candidates = [
    os.getenv("MODEL_PATH", "best.pt"),
    "runs/detect/train/weights/best.pt",
    "runs/detect/train2/weights/best.pt",
]
MODEL_PATH = next((candidate for candidate in _candidates if Path(candidate).is_file()), _candidates[0])


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #351313, #8f2424);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: .45rem 0 0 0; color: #ffe4e4;}
    .metric-card {
        padding: 1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        min-height: 105px;
    }
    .metric-label {font-size: .85rem; opacity: .75;}
    .metric-value {font-size: 1.7rem; font-weight: 700;}
    div[data-testid="stDownloadButton"] button {width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🔥 FireGuard AI</h1>
        <p>AI-powered fire detection using your custom-trained YOLO11 model</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading your trained YOLO model...")
def load_model(model_path: str):
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place your trained best.pt file in the project folder "
            "or set the MODEL_PATH environment variable."
        )
    return YOLO(model_path)


try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.info(
        "Copy your trained weights file (best.pt) into the same folder "
        "as app.py, then restart the application."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Detection settings")
image_confidence = st.sidebar.slider(
    "Image confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.40,
    step=0.05,
    help="Your Colab notebook uses conf=0.60 for image prediction.",
)
video_confidence = st.sidebar.slider(
    "Video confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Your Colab notebook uses conf=0.25 for video prediction.",
)
iou_threshold = st.sidebar.slider(
    "IoU threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
)
image_size = st.sidebar.select_slider(
    "Inference image size",
    options=[320, 416, 512, 640, 768, 896, 1024],
    value=640,
)
max_detections = st.sidebar.number_input(
    "Maximum detections per frame",
    min_value=1,
    max_value=300,
    value=100,
    step=10,
)
st.sidebar.caption("Image and video confidence values match your Colab notebook defaults.")
st.sidebar.divider()
st.sidebar.caption(f"Loaded weights: {MODEL_PATH}")
st.sidebar.caption(f"Model classes: {model.names}")


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def get_detection_summary(result):
    """Return detection count and confidence values for one frame/image."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0, []

    confidences = boxes.conf.detach().cpu().numpy().tolist()
    return len(confidences), confidences


def get_class_names(result):
    """Get detected class names from the model result."""
    names = result.names
    detected = []
    if result.boxes is not None:
        for class_id in result.boxes.cls.detach().cpu().numpy().astype(int):
            detected.append(names.get(int(class_id), str(class_id)))
    return detected


def render_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    # Ultralytics expects OpenCV-style BGR when the source is a NumPy array.
    # PIL provides RGB, so convert before inference.
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    result = model.predict(
        source=image_bgr,
        conf=image_confidence,
        iou=iou_threshold,
        imgsz=image_size,
        max_det=int(max_detections),
        verbose=False,
    )[0]

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    count, confidences = get_detection_summary(result)
    return image, annotated_rgb, result, count, confidences


def process_video(uploaded_file, progress_bar, status_text):
    """
    Process a video frame-by-frame to avoid storing all YOLO results in RAM.
    Output is written as MP4 using mp4v codec.
    """
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"

    input_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path = input_file.name
    output_path = output_file.name
    input_file.close()
    output_file.close()

    try:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open this video. Try another video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Could not read the video's dimensions.")

        # Limit output dimensions to a maximum of 1920 while preserving aspect ratio.
        scale = min(1.0, 1920 / max(width, height))
        out_width = max(2, int(width * scale))
        out_height = max(2, int(height * scale))
        out_width -= out_width % 2
        out_height -= out_height % 2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path, fourcc, fps, (out_width, out_height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not initialize the output video writer.")

        frame_count = 0
        total_detections = 0
        confidence_values = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            if scale != 1.0:
                frame = cv2.resize(frame, (out_width, out_height))
            elif frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(frame, (out_width, out_height))

            result = model.predict(
                source=frame,
                conf=video_confidence,
                iou=iou_threshold,
                imgsz=image_size,
                max_det=int(max_detections),
                verbose=False,
            )[0]

            annotated_frame = result.plot()
            writer.write(annotated_frame)

            count, confs = get_detection_summary(result)
            total_detections += count
            confidence_values.extend(confs)
            frame_count += 1

            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.write(
                    f"Processed {frame_count:,} / {total_frames:,} frames"
                )
            else:
                status_text.write(f"Processed {frame_count:,} frames")

        cap.release()
        writer.release()

        if frame_count == 0:
            raise ValueError("No readable frames were found in the uploaded video.")

        return (
            output_path,
            frame_count,
            total_detections,
            confidence_values,
            fps,
        )

    except Exception:
        # Remove incomplete output on failure.
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------
tab_image, tab_video, tab_about = st.tabs(
    ["🖼️ Image Detection", "🎥 Video Detection", "ℹ️ About"]
)

with tab_image:
    st.subheader("Detect fire in an image")
    st.write("Upload an image to identify visible fire regions.")

    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader",
    )

    if image_file is not None:
        if st.button("🔍 Detect Fire", type="primary", key="detect_image"):
            try:
                with st.spinner("Analyzing image..."):
                    original, annotated, result, count, confs = process_image(
                        image_file
                    )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original image**")
                    st.image(original, use_container_width=True)
                with col2:
                    st.markdown("**Detection result**")
                    st.image(annotated, use_container_width=True)

                st.divider()
                st.markdown(f"**Model class labels:** `{result.names}`")
                m1, m2, m3 = st.columns(3)
                with m1:
                    render_metric("Detected objects", count)
                with m2:
                    render_metric(
                        "Average confidence",
                        f"{np.mean(confs) * 100:.1f}%" if confs else "—",
                    )
                with m3:
                    render_metric(
                        "Detection status",
                        "Fire detected" if count else "No detection",
                    )

                if count:
                    st.success(f"{count} detection(s) found.")
                    rows = []
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i].item())
                        rows.append(
                            {
                                "Class": result.names.get(class_id, str(class_id)),
                                "Confidence": round(float(boxes.conf[i].item()), 4),
                            }
                        )
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.warning(
                        "No objects were detected. Try a confidence threshold of 0.10–0.20, "
                        "check that the loaded weights are your trained best.pt, and test "
                        "an image similar to the training dataset."
                    )

                image_bytes = cv2.imencode(
                    ".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                )[1].tobytes()
                st.download_button(
                    "⬇️ Download annotated image",
                    data=image_bytes,
                    file_name="fire_detection_result.jpg",
                    mime="image/jpeg",
                    type="primary",
                )

            except Exception as exc:
                st.error(f"Image processing failed: {exc}")


with tab_video:
    st.subheader("Detect fire in a video")
    st.write(
        "Upload a video to process it frame-by-frame and generate an annotated video."
    )

    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if video_file is not None:
        st.video(video_file)

        if st.button("🎬 Process Video", type="primary", key="detect_video"):
            progress = st.progress(0)
            status = st.empty()

            try:
                with st.spinner("Processing video. Larger videos may take time..."):
                    (
                        output_path,
                        frame_count,
                        total_detections,
                        confs,
                        fps,
                    ) = process_video(video_file, progress, status)

                st.success("Video processing completed.")
                m1, m2, m3 = st.columns(3)
                with m1:
                    render_metric("Frames processed", f"{frame_count:,}")
                with m2:
                    render_metric("Total detections", f"{total_detections:,}")
                with m3:
                    render_metric(
                        "Average confidence",
                        f"{np.mean(confs) * 100:.1f}%" if confs else "—",
                    )

                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                st.markdown("**Processed video**")
                st.video(video_bytes)

                st.download_button(
                    "⬇️ Download processed video",
                    data=video_bytes,
                    file_name="fire_detection_output.mp4",
                    mime="video/mp4",
                    type="primary",
                )

                os.remove(output_path)

            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
                st.caption(
                    "If the video is very large, try a shorter or lower-resolution clip."
                )


with tab_about:
    st.subheader("About FireGuard AI")
    st.markdown(
        """
        **FireGuard AI** is an object-detection application built with
        Streamlit and a custom-trained Ultralytics YOLO11 model.

        **Capabilities**
        - Image-based fire detection with bounding boxes.
        - Video-based, frame-by-frame fire detection.
        - Adjustable confidence and IoU thresholds.
        - Detection counts and confidence statistics.
        - Downloadable annotated outputs.

        **Model details**
        - Architecture: YOLO11s
        - Training image size: 640
        - Training epochs in the notebook: 100
        - Weights: `best.pt`

        **Important:** This is an AI-assisted detection tool, not a certified
        fire alarm or emergency-response system. It may miss fires or produce
        false detections. Never rely on it as the sole means of fire safety.
        """
    )

st.divider()
st.caption(
    "FireGuard AI • YOLO11 Fire Detection • Verify detections before taking action."
)
