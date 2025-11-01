# Yolo Vision Hub

A comprehensive computer vision project featuring face recognition, object detection, and image processing algorithms. This project integrates multiple vision tasks into a unified Streamlit web application.

## 📋 Project Overview

This repository contains three main tasks, each demonstrating different computer vision techniques:

1. **Face Recognition System** (Folder `1/`)
2. **YOLO Object Detection** (Folder `2/`)
3. **Image Processing Algorithms** (Folder `3/`)

## 🚀 Features

### Task 1: Face Recognition System (`1/`)

A complete face recognition pipeline using OpenCV's YuNet face detector and SFace face recognizer, combined with an SVM classifier.

#### Components:
- **Buoc1 (`1/Buoc1/get_face.py`)**: Face detection and extraction from video stream
  - Real-time face detection from IP camera
  - Automatic face alignment and cropping
  - Saves detected faces to dataset folder
  
- **Buoc2 (`1/Buoc2/Training.py`)**: Model training
  - Extracts 128-dimensional face embeddings using SFace
  - Trains LinearSVC classifier on face embeddings
  - Saves trained model (`svc.pkl`) for recognition
  
- **Buoc3 (`1/Buoc3/predict.py`)**: Real-time face recognition
  - Live face recognition from video stream
  - Displays recognized person names in real-time
  - Visualizes face landmarks and bounding boxes

#### Models Used:
- `face_detection_yunet_2023mar.onnx`: Face detection model
- `face_recognition_sface_2021dec.onnx`: Face recognition/embedding model
- `svc.pkl`: Trained SVM classifier

#### Dataset:
- Face images organized by person in `1/image/` directory
- Currently includes: DaiLong, MinhHieu, MinhHoang, VanThang, caothang

#### Usage:
```bash
# Step 1: Collect face images
cd 1/Buoc1
python get_face.py --video http://192.168.0.100:8080/video

# Step 2: Train the classifier
cd ../Buoc2
python Training.py

# Step 3: Run face recognition
cd ../Buoc3
python predict.py --video http://192.168.0.101:8080/video
```

---

### Task 2: YOLO Object Detection (`2/`)

YOLO v8-based object detection system for fruit recognition.

#### Features:
- **Training Script (`2/train_yolo.py`)**: Custom YOLO model training
- **GUI Application (`2/yolo_detector_gui.py`)**: Tkinter-based desktop application
  - Image upload and prediction
  - Real-time object detection with bounding boxes
  - Confidence score display

#### Classes Detected:
1. SauRieng (Durian)
2. Tao (Apple)
3. ThanhLong (Dragon Fruit)
4. Chuoi (Banana)
5. Kiwi

#### Dataset:
- Training data: `2/data/TraiCayx5-640x640_OK/`
  - Train set: 288 images
  - Validation set: 72 images
- Raw images: `2/data/TraiCayScratch/`
- Model: `2/model/best.onnx`

#### Usage:
```bash
# Train YOLO model
cd 2
python train_yolo.py

# Run GUI application
python yolo_detector_gui.py
```

---

### Task 3: Image Processing Algorithms (`3/`)

Comprehensive collection of image processing algorithms organized into three chapters.

#### Chapter 3: Basic Image Enhancement (`3/chapter3.py`)
- **Point Operations**:
  - Negative transformation
  - Logarithmic transformation
  - Power-law (gamma) transformation
  - Piecewise linear transformation
  
- **Histogram Processing**:
  - Histogram visualization
  - Histogram equalization (grayscale & color)
  - Local histogram equalization
  - Histogram statistics-based enhancement
  
- **Spatial Filtering**:
  - Smoothing filters (Box, Gaussian)
  - Median filter
  - Sharpening filters (Laplacian, Unsharp masking)
  - Gradient computation

#### Chapter 4: Frequency Domain Filtering (`3/chapter4.py`)
- FFT spectrum visualization
- Frequency domain filtering
- Moire pattern removal
- Motion blur simulation
- Interference removal

#### Chapter 9: Morphological Operations (`3/chapter9.py`)
- Erosion
- Dilation
- Boundary extraction
- Contour detection and visualization

#### Usage:
```bash
cd 3
python test_c3_4_9.py
```

---

## 🌐 Streamlit Web Application

A unified web interface (`streamlit/app.py`) that integrates all three tasks:

### Features:
1. **Face Recognition** (`Nhận diện khuôn mặt`)
   - Image upload
   - Video upload
   - Real-time webcam streaming (WebRTC)

2. **Object Detection** (`Nhận diện đối tượng`)
   - Fruit detection using YOLO model
   - Real-time prediction with confidence scores

3. **Image Processing** (`Xử lý ảnh số`)
   - Interactive UI for all Chapter 3, 4, and 9 operations
   - Before/after image comparison
   - Export processed images

4. **Handwriting OCR** (`Đọc chữ viết tay`)
   - Handwritten text recognition using LSTM-based model

### Running the Streamlit App:
```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## 📁 Project Structure

```
Yolo-Vision-Hub/
├── 1/                          # Face Recognition Task
│   ├── Buoc1/                  # Face collection
│   │   ├── get_face.py
│   │   └── get_face - Copy.py
│   ├── Buoc2/                  # Training
│   │   └── Training.py
│   ├── Buoc3/                  # Prediction
│   │   └── predict.py
│   ├── image/                  # Face dataset
│   │   ├── caothang/
│   │   ├── DaiLong/
│   │   ├── MinhHieu/
│   │   ├── MinhHoang/
│   │   └── VanThang/
│   └── model/                  # Pre-trained models
│       ├── face_detection_yunet_2023mar.onnx
│       ├── face_recognition_sface_2021dec.onnx
│       └── svc.pkl
│
├── 2/                          # YOLO Object Detection
│   ├── data/                   # Training dataset
│   │   ├── TraiCayScratch/     # Raw images
│   │   └── TraiCayx5-640x640_OK/  # Processed dataset
│   ├── model/                  # YOLO model
│   │   └── best.onnx
│   ├── train_yolo.py          # Training script
│   ├── train_yolo_v8n(2).ipynb
│   ├── yolo_detector_gui.py   # GUI application
│   └── yolo2.py
│
├── 3/                          # Image Processing
│   ├── chapter3.py            # Basic enhancement
│   ├── chapter4.py            # Frequency domain
│   ├── chapter9.py            # Morphology
│   ├── test_c3_4_9.py        # Test script
│   └── image_test/            # Test images
│
├── 4/                          # OCR Model Training
│   ├── data_test/
│   ├── model/
│   │   └── best_model.h5
│   └── train.ipynb
│
└── streamlit/                  # Web application
    ├── app.py                 # Main Streamlit app
    ├── chapter3.py, chapter4.py, chapter9.py
    ├── face_recognition_utils.py
    ├── video_processor.py
    ├── stream_processor.py
    ├── ocr_utils.py
    ├── requirements.txt
    └── model/                 # All models
        ├── best_model.h5      # OCR model
        ├── best.onnx          # YOLO model
        ├── face_detection_yunet_2023mar.onnx
        ├── face_recognition_sface_2021dec.onnx
        └── svc.pkl            # Face recognition classifier
```

---

## 🛠️ Dependencies

### Core Libraries:
- `opencv-python` / `opencv-contrib-python` - Computer vision operations
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning (SVM classifier)
- `ultralytics` - YOLO v8 implementation
- `joblib` - Model serialization

### For Streamlit App:
- `streamlit` - Web application framework
- `streamlit-webrtc` - Real-time video streaming
- `keras` / `tensorflow` - Deep learning (OCR model)

### Installation:
```bash
# Install core dependencies
pip install opencv-python opencv-contrib-python numpy scikit-learn ultralytics joblib

# Install Streamlit dependencies
cd streamlit
pip install -r requirements.txt
```

---

## 📝 Usage Examples

### Face Recognition Pipeline:
1. **Collect face data**: Run `1/Buoc1/get_face.py` with your IP camera
2. **Train model**: Execute `1/Buoc2/Training.py` to train the SVM classifier
3. **Recognize faces**: Use `1/Buoc3/predict.py` for real-time recognition

### YOLO Fruit Detection:
1. **Train model**: Use `2/train_yolo.py` or the Jupyter notebook
2. **Run GUI**: Launch `2/yolo_detector_gui.py` for desktop application
3. **Use in Streamlit**: Access through the web interface

### Image Processing:
1. Import functions from `chapter3.py`, `chapter4.py`, or `chapter9.py`
2. Use the Streamlit web interface for interactive processing
3. Or run `test_c3_4_9.py` for batch processing

---

## 🎯 Key Technologies

- **Face Recognition**: OpenCV DNN (YuNet + SFace) + SVM
- **Object Detection**: YOLO v8 (Ultralytics)
- **Image Processing**: OpenCV + NumPy
- **Web Interface**: Streamlit + WebRTC
- **OCR**: LSTM-based neural network (Keras/TensorFlow)

---

## 📄 License

This project is for educational and research purposes.

---

## 👤 Author

Demo by HieuDuong

---

## 🔗 Model Sources

- **YuNet Face Detector**: [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- **SFace Face Recognizer**: [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- **YOLO v8**: [Ultralytics](https://github.com/ultralytics/ultralytics)

---

## 📌 Notes

- IP camera URLs need to be configured in the scripts
- Model paths may need adjustment based on your system
- Ensure all model files are present in the respective `model/` directories
- For best results, use high-quality images and proper lighting conditions

