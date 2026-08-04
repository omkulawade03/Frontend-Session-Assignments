# Session 27 (AIML) – Assignment

## Student Information

- **Name:** Om Prashant Kulawade
- **College:** Zeal Polytechnic, Narhe, Pune
- **Branch:** Artificial Intelligence and Machine Learning (AIML)
- **Domain:** Artificial Intelligence and Machine Learning (AIML)

---

# Assignment Title

**DBSCAN Clustering and Principal Component Analysis (PCA)**

---

# Objective

The objective of this assignment is to understand density-based clustering using **DBSCAN**, compare it with **K-Means**, and apply **Principal Component Analysis (PCA)** for dimensionality reduction and visualization of high-dimensional datasets.

---

# Assignment Questions

### Q1. Creating Non-Linear Dataset
- Generate a synthetic moon dataset using `make_moons()`.
- Convert it into a Pandas DataFrame.
- Display the first 10 rows and dataset shape.

### Q2. Scaling the Data
- Apply `StandardScaler` to normalize the dataset.
- Store the scaled data in `X_scaled`.

### Q3. K-Means on Non-Linear Data
- Apply K-Means clustering (`n_clusters=2`).
- Predict cluster labels.
- Visualize the clustering results.

### Q4. DBSCAN Clustering
- Apply DBSCAN with:
  - `eps = 0.3`
  - `min_samples = 5`
- Predict cluster labels.
- Visualize the clustering results.

### Q5. Comparing K-Means vs DBSCAN
- Compare both clustering methods.
- Explain why DBSCAN performs better on non-linear datasets and handles noise effectively.

### Q6. Tuning DBSCAN Parameters
- Experiment with different `eps` values:
  - 0.2
  - 0.3
  - 0.4
  - 0.5
- Count clusters and noise points.
- Identify the best `eps` value.

### Q7. Creating High-Dimensional Dataset
- Generate a synthetic dataset using `make_blobs()`.
- Create:
  - 500 samples
  - 6 features
  - 4 centers
- Scale the dataset.

### Q8. Applying PCA
- Reduce the dataset to **2 Principal Components**.
- Create a DataFrame with **PC1** and **PC2**.
- Display the explained variance ratio.

### Q9. Visualizing PCA Results
- Plot the two principal components.
- Color the points using the original cluster labels.
- Explain the meaning of the explained variance ratio.

### Q10. Mini Project – Complete Pipeline
Perform the complete machine learning workflow:
1. Load a real-world dataset
2. Select numerical features
3. Scale the data
4. Apply DBSCAN
5. Apply PCA
6. Visualize the final results

---

# Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Datasets Used

### Synthetic Datasets
- Moon Dataset (`make_moons`)
- Blob Dataset (`make_blobs`)

### Real Dataset
- Wine Dataset (Scikit-learn)
or
- Iris Dataset (Scikit-learn)

---

# Project Structure

```
Session_27_AIML_Assignment/
│
├── Session_27_Assignment.ipynb
├── README.md
│
├── data/
   ├── moon_dataset.csv
   ├── blobs_dataset.csv
   └── wine_dataset.csv
---

# Expected Outcome

After completing this assignment, students will be able to:

- Understand the working of DBSCAN clustering.
- Compare K-Means and DBSCAN algorithms.
- Tune DBSCAN parameters effectively.
- Apply PCA for dimensionality reduction.
- Visualize high-dimensional datasets.
- Build a complete clustering and PCA pipeline using a real-world dataset.

---

# Submitted By

**Om Prashant Kulawade**

Artificial Intelligence and Machine Learning (AIML)

Zeal Polytechnic, Narhe, Pune

---

**Session 27 – AIML Assignment**