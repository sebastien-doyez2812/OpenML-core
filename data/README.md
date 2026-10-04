# OpenML Core — Dataset Format Guide

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/SDoyez/Augmented_Dubai-Satellite_Image_segmentation)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

This guide describes the expected directory structure and labeling formats for dataset integration.

---

## 📁 Directory Structure

Your dataset must be split into `train` and `val` directories with the following structure:

```text
data/
├── train/
│   ├── imgs/     # Raw input images (.png, .jpg)
│   └── labels/   # Ground truth segmentation masks
└── val/
    ├── imgs/     # Raw input images (.png, .jpg)
    └── labels/   # Ground truth segmentation masks
```

---

## 🏷️ Label Encoding Formats

Depending on your task complexity, ground truth labels should follow one of these formats:

### 1. One-Hot / Class Indexing (Single Class per Pixel)
If each pixel belongs strictly to a **single class** among $C$ possible categories, the label mask is a 2D matrix of shape `[H, W]` containing class index integers ($0$ to $C-1$).

### 2. Multi-Label Encoding (Multiple Classes per Pixel)
If a pixel can simultaneously belong to **multiple classes** (e.g., a *car* overlapping with a *road*), the label mask must be a 3D tensor of shape `[H, W, C]`.

#### Multi-Label Vector Example ($C = 8$ classes)
| Class | Road | Sky | Building | Car | Plane | Human | Tree | Pet |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Active** | **1** | 0 | 0 | **1** | 0 | 0 | 0 | 0 |

> **Note:** In this case, the pixel vector `[1, 0, 0, 1, 0, 0, 0, 0]` marks both **Road** and **Car**.

---

## 🚀 Quickstart & Example Dataset

You can test this layout directly using the **Augmented Dubai Satellite Image Segmentation** dataset available on Hugging Face Hub.

### Option A: Clone via Git LFS
```bash
git clone https://huggingface.co/datasets/SDoyez/Augmented_Dubai-Satellite_Image_segmentation
```

### Option B: Load via Python `datasets`
```python
from datasets import load_dataset

# Load dataset from Hugging Face Hub
ds = load_dataset("SDoyez/Augmented_Dubai-Satellite_Image_segmentation")
```