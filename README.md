# Super Resolution of Thermal Images for Object Detection
## Introduction

This project focuses on the integration of **super-resolution** and **object detection** techniques to improve image quality and enhance detection capabilities. These techniques, especially applied to thermal images from road contexts, can improve safety and efficiency in complex urban scenarios.

Specifically, a **Real-ESRGAN** super-resolution network was used to improve the resolution of thermal images. Then, a **YOLO** object detection network was applied. Both models were refined via **fine-tuning** on a real thermal image dataset, provided by the **FLIR** dataset.

---

## Pipeline Structure

1. **Super-Resolution with Real-ESRGAN Fine-Tuned**
- Improves the resolution of thermal images.
- Reduces the over-smoothing effect via fine-tuning on real thermal images.

2. **Patch Splitting**
- The super-resolution images are split into **overlapping patches** to improve the detection of small objects and distant details.

3. **Object Recognition with YOLO Fine-Tuned**
- A YOLO model trained specifically on thermal images is used for object detection.

4. **Results Merge**
- The results from the patches and the original image are merged using **Non Maximum Suppression (NMS)** to obtain bounding boxes and consolidated classes.

---

## Dataset

- **FLIR Thermal Dataset**
- Contains thermal street images used for fine-tuning the models.
- Training set: 8,862 8-bit images.

---

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- Real-ESRGAN
- YOLOv11 (with YOLO Patch-Based Inference)

## BibTeX

    @InProceedings{megli2024sr4od,
        author    = {David megli},
        title     = {Super Resolution of Thermal Images for Object Detection},
        date      = {2024}
    }

## 📧 Contact

If you have any question, please email `david.megli@outlook.com`