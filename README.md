# 🔥 FireGuard AI – Real-Time Fire Detection System

An AI-powered Fire Detection System built using **YOLO11s (Ultralytics)** and **Streamlit** to detect fire in images and videos. The system uses a custom-trained deep learning object detection model to identify fire, display bounding boxes with confidence scores, and generate annotated outputs.

## 🚀 Project Overview

Fire accidents can cause significant damage to human lives, property, and the environment. Early fire detection plays an important role in reducing these risks.

**FireGuard AI** uses computer vision and deep learning to detect fire from uploaded images and videos. It provides an interactive web interface where users can upload media, adjust detection thresholds, view detection results, and download annotated outputs.

## ✨ Key Features

- 🔥 **AI-Based Fire Detection:** Detects fire using a custom-trained YOLO11s model.
- 🖼️ **Image Detection:** Supports JPG, JPEG, and PNG images.
- 🎥 **Video Detection:** Supports MP4, AVI, and MOV video formats.
- 🎯 **Bounding Boxes:** Highlights detected fire regions.
- 📊 **Confidence Scores:** Displays confidence scores for detected objects.
- ⚙️ **Adjustable Thresholds:** Allows users to customize detection confidence and IoU thresholds.
- 📈 **Detection Statistics:** Displays useful detection information.
- 📥 **Download Results:** Download annotated images and processed videos.
- 🌐 **Interactive Web Interface:** Built using Streamlit.

## 🧠 Model Architecture

This project uses **YOLO11s (You Only Look Once)** from Ultralytics.

YOLO is a real-time object detection architecture that identifies objects and predicts their locations in a single forward pass.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11s |
| Framework | Ultralytics |
| Task | Object Detection |
| Training Epochs | 100 |
| Image Size | 640 × 640 |
| Best Model Weights | best.pt |

The trained model weights are saved as `best.pt` and used for inference.

## 🛠️ Technologies Used

- Python
- YOLO11s
- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Streamlit
- Pillow

## 📂 Project Structure

```text
FireGuard-AI/
│
├── app.py                  # Streamlit application
├── best.pt                 # Trained YOLO11s model
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation

```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FireGuard-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd FireGuard-AI
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

The project uses the following Python libraries:

```text
streamlit
ultralytics
opencv-python-headless
numpy
pillow
torch
torchvision
imageio-ffmpeg
```

## 🔍 How It Works

1. **Upload Media:** Upload an image or video through the Streamlit interface.
2. **Model Loading:** The application loads the custom-trained YOLO11s model using `best.pt`.
3. **Preprocessing:** The uploaded media is processed using OpenCV.
4. **Object Detection:** YOLO11s identifies fire regions and generates bounding boxes.
5. **Visualization:** Detection results and confidence scores are displayed.
6. **Download:** Download the annotated output for further use.

## 🎯 Applications

- Early fire detection and monitoring.
- Industrial safety monitoring.
- Forest fire surveillance.
- Building and infrastructure safety.
- Video-based fire monitoring systems.

## 🔮 Future Improvements

- Real-time CCTV camera integration.
- Automated fire alerts and notifications.
- Integration with IoT-based fire alarm systems.
- Edge-device deployment.
- Improved detection performance in challenging lighting conditions.

## 👨‍💻 Author

**Sujith Karnati**

B.Tech – Computer Science and Engineering (AI & ML)

Interested in Artificial Intelligence, Machine Learning, Deep Learning, and Computer Vision.

## ⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub!

---

**FireGuard AI – Making Fire Detection Smarter with Artificial Intelligence.** 🔥
