# Assignment 26 – K-Means Clustering

## Student Information

- **Name:** Om Prashant Kulawade
- **College:** Zeal Polytechnic, Narhe, Pune
- **Branch:** Artificial Intelligence and Machine Learning (AIML)
- **Domain:** Artificial Intelligence and Machine Learning (AIML)

---

## Assignment Overview

This assignment demonstrates the implementation of **K-Means Clustering**, an unsupervised machine learning algorithm used to group similar data points into clusters. It covers synthetic data generation, data preprocessing, cluster visualization, the Elbow Method, and clustering on real-world datasets.

---

## Objectives

- Generate and analyze synthetic datasets.
- Apply feature scaling using StandardScaler.
- Perform K-Means clustering.
- Visualize clusters using Seaborn and Matplotlib.
- Determine the optimal number of clusters using the Elbow Method.
- Apply clustering techniques on a real-world dataset.
- Build a complete clustering pipeline for customer segmentation.

---

## Assignment Questions

### Q1. Understanding & Dataset Creation
- Generate a synthetic dataset using `make_blobs()`.
- Convert the dataset into a Pandas DataFrame.
- Display the first 10 rows and dataset shape.

### Q2. Data Scaling
- Apply Standard Scaling using `StandardScaler`.
- Store the scaled dataset in `X_scaled`.

### Q3. Basic K-Means Clustering
- Train a K-Means model with **3 clusters**.
- Predict cluster labels.
- Add cluster labels to the DataFrame.

### Q4. Cluster Visualization
- Create a scatter plot using Seaborn.
- Visualize clusters using different colors.

### Q5. Elbow Method
- Compute inertia values for **K = 1 to 10**.
- Plot the Elbow Curve.
- Identify the optimal number of clusters.

### Q6. Final K-Means Model
- Train the final model using the optimal K.
- Predict cluster labels.
- Display the number of samples in each cluster.

### Q7. Final Cluster Visualization
- Plot the final clusters.
- Display cluster centroids.

### Q8. Real Dataset Application
- Load a real-world dataset (Iris or Mall Customers).
- Select numerical features.
- Scale the data.
- Apply K-Means clustering.
- Display cluster counts.

### Q9. Elbow Method on Real Dataset
- Apply the Elbow Method.
- Select the optimal K.
- Train the final model.
- Visualize clusters or display cluster sizes.

### Q10. Mini Project – Complete Clustering Pipeline
- Load and explore the dataset.
- Perform feature scaling.
- Apply the Elbow Method.
- Train the final K-Means model.
- Visualize the clustering results.

---

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Dataset Used

### Synthetic Dataset
- Generated using `make_blobs()` from Scikit-learn.

### Real Dataset
- Iris Dataset
or
- Mall Customers Dataset

---

## Project Structure

```
Assignment_26/
│
├── Assignment_26.ipynb
├── README.md
├── data/
   └── Mall_Customers.csv


## Expected Outcome

After completing this assignment, students will be able to:

- Understand the K-Means clustering algorithm.
- Perform feature scaling.
- Determine the optimal number of clusters using the Elbow Method.
- Visualize clustering results.
- Apply clustering techniques to real-world datasets.

---

## Submitted By

**Om Prashant Kulawade**

Artificial Intelligence and Machine Learning (AIML)

Zeal Polytechnic, Narhe, Pune

---

**Assignment 26 – AIML**