# 📊 K-Means Clustering with Elbow and Silhouette Analysis

This project demonstrates **K-Means clustering** on a synthetic dataset generated using `make_blobs`. It includes techniques for choosing the optimal number of clusters using:

- 📐 **Elbow Method**
- 🔍 **Silhouette Score**
- 📉 **Knee Locator**

---

## 🚀 Features

- Synthetic 2D dataset using `sklearn.datasets.make_blobs`
- Dataset split into train and test sets
- Manual implementation of the **Elbow method** with WCSS
- Use of **KneeLocator** to automate elbow point detection
- **Silhouette score** analysis for cluster quality evaluation
- Cluster visualization with `matplotlib`

---

## 🛠️ Libraries Used

```bash
numpy  
pandas  
matplotlib  
scikit-learn  
kneed
