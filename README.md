# 🧠 Image Clustering using SCAN with Visual Explainability (LIME)

This project implements unsupervised image clustering using the SCAN (Semantic Clustering by Adopting Nearest neighbors) framework and enhances the clustering results with visual explanations using the XAI technique LIME (Local Interpretable Model-agnostic Explanations).

The goal is not only to cluster images effectively but also to understand which features contribute most to cluster assignments.

---

## 🎯 Objective

The main objectives of this project are:

- Perform unsupervised image clustering using the SCAN framework
- Apply clustering on the CIFAR100-20 dataset
- Evaluate clustering performance using standard metrics
- Provide visual explainability of clustering results using LIME
- Identify and visualize the most important features contributing to cluster assignments

Reference Paper: SCAN – Semantic Clustering by Adopting Nearest neighbors

---

## 👨‍💻 My Contribution

My primary contribution to this project was implementing Explainable AI (XAI) using LIME to interpret clustering results.

Specifically, I worked on:

- Applying LIME to clustered image outputs
- Generating visual explanations for cluster assignments
- Identifying important image regions influencing clustering decisions
- Creating prototype visualizations and heatmaps
- Enhancing model transparency and interpretability

This improves trust and understanding of unsupervised clustering models.

---

## ⚙️ Installation

This project uses PyTorch and several supporting libraries.

Create a conda environment and install dependencies:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib scipy scikit-learn
conda install faiss-gpu
conda install pyyaml easydict
conda install termcolor
conda install -c conda-forge grad-cam
pip install lime
```

Alternatively, install using:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

We use the CIFAR100-20 dataset, which groups CIFAR-100 classes into 20 superclasses.

The trained models are included in the repository:

```
repository_eccv/cifar-20/scan/model.pth.tar
repository_eccv/cifar-20/selflabel/model.pth.tar
```

---

## 🧠 Training Overview

The SCAN training pipeline consists of:

1. Feature extraction using pretext training (SimCLR)
2. Finding nearest neighbors using feature similarity
3. Training clustering model using SCAN
4. Refining clusters using self-labeling

SCAN framework source: https://github.com/wvangansbeke/Unsupervised-Classification

---

## ▶️ Execution Steps

### Step 1: Find nearest neighbors

```bash
python find_k_nearest_neighbours.py \
--config_env configs/env.yml \
--config_exp configs/pretext/simclr_cifar20.yml
```

---

### Step 2: Cluster images using SCAN

```bash
python Cluster_img.py \
--n 30 \
--query 500 \
--config_exp configs/scan/scan_cifar20.yml \
--model repository_eccv/cifar-20/scan/model.pth.tar
```

---

### Step 3: Evaluate clustering performance

```bash
python eval_charts.py \
--query 30 \
--config_exp configs/scan/scan_cifar20.yml \
--model repository_eccv/cifar-20/scan/model.pth.tar
```

---

### Step 4: Apply LIME for Explainable AI (My Contribution)

```bash
python Lime.py \
--n 300 \
--query 500 \
--config_exp configs/scan/scan_cifar20.yml \
--model repository_eccv/cifar-20/scan/model.pth.tar \
--visualize_prototypes
```

This generates:

- Prototype image for each cluster
- LIME heatmaps showing important regions
- Visual explanation of clustering decisions

---

### Step 5: Visual Explanation for a Query Image

```bash
python scan_lime_explainability.py \
--query_image_path path/to/query_image.png \
--save_path path/to/save_results/
```

This produces:

- Cluster assignment
- LIME explanation heatmap
- Feature importance visualization

---

## 📊 Results

The clustering performance was evaluated using the following metrics:

- Accuracy (ACC)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Accuracy with Top-5 neighbors

---

## 📈 Visual Outputs

The project generates:

- Confusion Matrix
- Accuracy vs Number of Neighbors graph
- Cluster prototype visualizations
- LIME explanation heatmaps

These visualizations help understand both clustering performance and feature importance.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- SCAN Framework
- LIME (Explainable AI)
- FAISS (Nearest neighbor search)
- Scikit-learn
- Matplotlib

---

## 🚀 Key Features

- Unsupervised image clustering using SCAN
- Efficient nearest neighbor search using FAISS
- Explainable AI using LIME
- Visual heatmaps for interpretability
- Cluster prototype visualization

---

## 📌 Summary

This project combines state-of-the-art unsupervised clustering with Explainable AI to provide both accurate clustering and interpretable results.

The integration of LIME significantly improves transparency by highlighting image features responsible for cluster assignments.

---

## 👨‍💻 Authors

Team Project Contribution

Explainable AI (LIME) Implementation and Visual Explainability: Archana Yadav
SCAN Framework Integration and Clustering Pipeline: Venkatesh Date and Team

Original repository:
https://github.com/venkatesh281996/SCAN_XAI_LIME

