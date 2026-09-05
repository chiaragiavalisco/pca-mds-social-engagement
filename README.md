# Dimensionality Reduction on Social Media Engagement: PCA vs MDS Analysis

[![MATLAB](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the implementation, theoretical comparison, and experimental results for evaluating **Principal Component Analysis (PCA)** and **Multidimensional Scaling (MDS)** applied to social media marketing performance.

The study investigates engagement data collected from the official Facebook pages of **9 Thai retail sellers** operating in the fashion and cosmetics sectors. By projecting high-dimensional interaction metrics into a low-dimensional latent space ($k = 2$), the analysis identifies latent marketing dynamics and highlights the **top 3 sellers** with dominant social management strategies.

---

## 🔬 Overview & Research Aim

In social media marketing, user engagement is multidimensional, encompassing both broad visibility (shares, likes) and granular sentimental feedback (comments, specific emoji reactions). Evaluating brand performance across these heterogeneous dimensions is challenging due to scale disparities and collinearity.

The primary goals of this project are:
1. **Theoretical comparison**: Implementing both PCA (covariance/coordinate-based SVD) and Classical MDS (Euclidean distance matrix double-centering) from scratch without relying solely on high-level black-box toolboxes.
2. **Empirical verification**: Demonstrating the duality and equivalence between PCA and classical metric MDS under Euclidean distance.
3. **Retail performance clustering**: Identifying underlying performance patterns among the 9 Thai fashion and cosmetics retailers to single out the top performers in customer engagement.

---

## 📊 Dataset & Feature Engineering

The analyzed dataset records social engagement metrics across 9 Thai fashion & cosmetic retailers on Facebook. Each retailer is characterized across **8 interaction metrics**:

| Metric | Type | Description |
| :--- | :--- | :--- |
| `comments` | Engagement | Volume of user comments generated per post |
| `shares` | Virality | Number of times posts were shared |
| `likes` | Positive baseline | Standard thumbs-up reactions |
| `loves` | Sentimental | High-affinity positive emoji reactions |
| `wows` | Sentimental | Surprise / curiosity reactions |
| `hahas` | Sentimental | Humorous reactions |
| `sads` | Sentimental | Negative / sympathy reactions |
| `angrys` | Sentimental | Negative / controversy reactions |

### Normalization & Scaling
Because metrics such as `comments` often span orders of magnitude larger than specific emotions (e.g., `angrys`), features are standardized by their maximum-to-dispersion ratios:

$$\mathbf{X} = \left(\frac{\mathbf{MAX}}{\mathbf{SD}}\right)^T$$

where $\mathbf{MAX}$ and $\mathbf{SD}$ denote respectively the maximum observed values and standard deviations across items and $\mathbf{X} \in \mathbb{R}^{m \times n}$ ($m$ features and $n$ observations).

The matrix is mean-centered across observations prior to covariance decomposition:

$$\boldsymbol{\mu} = \frac{\mathbf{X} \mathbf{1}_n}{n}$$

$$\mathbf{\tilde{X}} = \mathbf{X} - \boldsymbol{\mu} \mathbf{1}_n^T$$

The sample covariance matrix is then computed as:

$$\mathbf{C}_X = \frac{\mathbf{\tilde{X}} \mathbf{\tilde{X}}^T}{n - 1}$$

---

## ⚙️ Methodology

### Principal Component Analysis (PCA)
- **Centering & Covariance**: The sample covariance matrix is constructed as:
  $$\mathbf{C}_X = \frac{1}{n - 1} \mathbf{	\tilde{X}}\mathbf{	\tilde{X}}^T$$
- **SVD Decomposition**:
  $$\mathbf{C}_X = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
- **Projection**: The data points are mapped onto the first $k = 2$ principal eigenvectors $\mathbf{U}_k$:
  $$\mathbf{Z} _{PCA}= \mathbf{U}_k^T \mathbf{	\tilde{X}}$$

### Classical Multidimensional Scaling (MDS)
- **Distance Matrix ($\mathbf{D}^{2}$)**: Pairwise squared Euclidean distances are computed between all observations:
  $$d_{ij}^2 = \|\mathbf{x}_i - \mathbf{x}_j\|^2$$
- **Double Centering**: Using centering matrix $\mathbf{H} = \mathbf{I} - rac{1}{n} \mathbf{1}\mathbf{1}^T$:
  $$\mathbf{B} = -\frac{1}{2} \mathbf{H} \mathbf{D}^{2} \mathbf{H}^T$$
- **Coordinate Recovery**:
  $$\mathbf{B} = \mathbf{W} \Lambda \mathbf{W}^T \implies \mathbf{Z}_{\text{MDS}} =  \Lambda_k^{1/2} \mathbf{W}_k^T$$

### Equivalence & Error Metrics
Both representations preserve identical relative configurations up to orthogonal rotations/reflections. The code computes:
- Relative PCA reconstruction error: $\frac{\Vert\mathbf{\tilde{X}} - \mathbf{U}_k \mathbf{U}_k^T \mathbf{\tilde{X}}\Vert_F}{\Vert\mathbf{\tilde{X}}\Vert_F}$
- Classical MDS Gram reconstruction error: $\frac{\Vert\mathbf{\tilde{X}}^T \mathbf{\tilde{X}} - \mathbf{Z}_{MDS}^T \mathbf{Z}_{MDS}\Vert_F}{\Vert\mathbf{\tilde{X}}^T \mathbf{\tilde{X}}\Vert_F}$
- Cross-method divergence: $\Vert \mathbf{Z}_{\text{PCA}} - \mathbf{Z}_{\text{MDS}}\Vert_F \approx 0$
---

## 📈 Key Findings & Business Insights

1. **Dimensionality Reduction & Explained Variance**:
   - The first two principal components capture the vast majority of cumulative variance ($>85\%$), proving that a 2D projection faithfully reflects the structure of the original 8-dimensional space.
2. **Feature Loadings (Load Plot)**:
   - **PC1 (Volume & Virality)** is heavily influenced by shares and comments, differentiating mass-reach viral content from static engagement.
   - **PC2 (Sentimental Depth & Reaction Bias)** captures emotional intensity, separating accounts relying on baseline likes from those driving proactive discussions and reactions (`love`, `wow`, `haha`).
3. **Identification of Top 3 Sellers**:
   - Out of the 9 Thai retailers analyzed, **three distinct sellers emerge as outliers along PC1/MD1 and PC2/MD2**:
     - They demonstrate a balanced portfolio of high viral distribution (`shares`), active discussion (`comments`), and authentic sentiment (`love`/`wow`).
     - The remaining sellers cluster close to the origin, indicating low relative reach and low active engagement across their posting schedules.

---

## 📂 Repository Structure

```text
├── SecondAssignment.m      # Main MATLAB script (PCA, MDS, Scree plot, Loadings, Scatter)
├── README.md               # Project documentation and summary
├── data/
│   └── thai_retail_fb.mat  # (Optional) Pre-formatted workspace data
└── figures/                # Output plots
    ├── pca_2d_scatter.png  # 2D projection on PC1 vs PC2
    ├── mds_2d_scatter.png  # 2D projection on MD1 vs MD2
    ├── load_plot.png       # Feature loadings for engagement metrics
    └── scree_plot.png      # Eigenvalue scree plot and explained variance
```

---

## 🚀 Getting Started & Usage

### Prerequisites
- MATLAB (R2018b or later recommended) or GNU Octave (v6.0+).
- Statistics and Machine Learning Toolbox (optional, the script implements PCA/MDS natively via `svd`).

### Running the Script
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Open MATLAB and run `SecondAssignment.m`:
   ```matlab
   run('SecondAssignment.m')
   ```
3. The script will output:
   - Numerical projection matrices `z` (PCA) and `z1` (MDS)
   - Relative errors (`errrelPCA`, `errrelMDS`)
   - Scree plot cumulative variance
   - Figures 1 to 6 (1D and 2D scatter plots, feature loading vectors, scree plots)

---

## 📄 References & Citation

If you use this work or codebase for academic purposes, please cite:

```bibtex
@article{thai_retail_pca_mds,
  title={Dimensionality Reduction and Retail Social Management Analysis: A Comparative Study of PCA and MDS on Facebook Engagement},
  author={Your Name},
  year={2024}
}
```
