# Monocular Depth Estimation (MDE) using ResNet-50 & U-Net

This repository contains the implementation of a dense-prediction deep learning network for Monocular Depth Estimation. Developed as part of an engineering internship at IIT Jodhpur, the model predicts the physical depth of a scene from a single 2D RGB image.

The architecture fuses a pre-trained ResNet-50 encoder with a custom U-Net style spatial decoder, optimized via a custom edge-aware loss function to preserve sharp object boundaries.

## Architecture Overview

* **Encoder:** Pre-trained ResNet-50 backbone. Extracts hierarchical feature maps at four distinct spatial resolutions.
* **Decoder:** Custom U-Net style upsampling blocks. Utilizes bilinear interpolation and spatial skip connections to concatenate high-resolution encoder features with upsampled representations, recovering lost spatial details.
* **Loss Function:** Standard L1 (Mean Absolute Error) combined with a **first-order finite difference** edge-aware penalty to prevent the "Safe Average" blurring phenomenon. To respect strict VRAM limits, this gradient calculation was engineered using native **tensor slicing** rather than standard 2D convolution matrices, achieving highly efficient, zero-parameter edge detection.

    `L_total = L1(y, y_hat) + λ(|∇x_y - ∇x_y_hat| + |∇y_y - ∇y_y_hat|)`

## Engineering & Hardware Constraints

Training a 23M+ parameter dense-prediction model on an NVIDIA RTX 5060 (8GB VRAM) required strict memory management. A two-phase training methodology was implemented to avoid Out-Of-Memory (OOM) exceptions and catastrophic forgetting:

* **Phase 1 (Frozen Backbone):** ResNet-50 weights frozen. U-Net decoder initialized and trained from scratch using a batch size of 8.
* **Phase 2 (End-to-End Fine-Tuning):** Entire network unfrozen. Optimizer states reset, learning rate reduced, and batch size strictly reduced to 4 to accommodate the massive gradient graph.

## Quantitative Results

Evaluated on the NYU Depth V2 validation split:

| Metric | Phase 1 (Frozen) | Phase 2 (End-to-End) |
| :--- | :--- | :--- |
| **RMSE** ↓ | 860.25 | 854.64 |
| **Strict Accuracy (δ < 1.25)** ↑ | 67.35% | 68.71% |
| **Loose Accuracy (δ < 1.25³)** ↑ | 98.37% | 98.62% |

## Installation & Setup

Clone the repository:

```bash
git clone https://github.com/Parth-Sharma-Dev/mde-pytorch-iitj.git

```

Install dependencies:

```bash
pip install torch torchvision numpy opencv-python matplotlib

```

Dataset Preparation:
Download the NYU Depth V2 dataset and extract it into the `data/` directory. Ensure the structure matches the dataloader paths:

```text
data/nyu_depth_v2/
├── train/
│   ├── rgb/
│   └── depth/
└── val/
    ├── rgb/
    └── depth/

```

## Usage

### Evaluation & Inference

To evaluate the model:

```bash
python eval.py

```

### Training

To initiate the two-phase training loop:

```bash
python train.py --batch_size 8 --epochs 170

```

## Future Scope

* **Vision Transformer (ViT) Integration:** Currently developing a Reassemble Module to replace the ResNet-50 backbone with a ViT-Base/16 architecture for superior global context processing.
* **Edge Deployment:** Porting the inference pipeline to ONNX/TensorRT for low-latency quantization, targeting intelligent infrastructure and real-time smart traffic management systems.

## Contributors

* **Parth Sharma** - Backend & AI-ML Engineering
* **Krishna Acharya**
* **Supervised by:** Prof. Avinash Sharma (IIT Jodhpur)
