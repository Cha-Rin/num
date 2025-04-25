# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 22:05:25 2025

@author: MS1
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load model safely
try:
    with open('kmeans_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Set the page config
st.set_page_config(page_title="K-Means Clustering App", layout="centered") 

# Set title
st.title("k-Means Clustering Visualizer")

# Display cluster centers
st.subheader("Example Data for Visualization")
st.markdown("This demo uses example data (2D) to illustrate clustering results.")

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict
y_kmeans = loaded_model.predict(X)
