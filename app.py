# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 22:05:25 2025

@author: MS1
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# โหลดโมเดล
try:
    with open('kmeans_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


st.set_page_config(page_title="K-Means Clustering App", layout="centered") 


st.title("k-Means Clustering Visualizer")


st.subheader("Example Data for Visualization")
st.markdown("This demo uses example data (2D) to illustrate clustering results.")


X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)


y_kmeans = loaded_model.predict(X)


fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')


centers = loaded_model.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='o', label='Centers')


ax.set_title("K-Means Clustering Results")
ax.legend()


st.pyplot(fig)

