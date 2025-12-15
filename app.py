import numpy as np
import streamlit as st

from prediction import predict

st.title("Classifying Iris Flowers")
st.markdown(
    "Toy model to classify iris flowers into setosa, versicolor, or virginica "
    "based on their sepal and petal dimensions."
)

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider("Sepal length (cm)", 4.0, 8.5, 5.8, step=0.1)
    sepal_w = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, step=0.1)

with col2:
    st.text("Petal characteristics")
    petal_l = st.slider("Petal length (cm)", 1.0, 7.5, 4.35, step=0.1)
    petal_w = st.slider("Petal width (cm)", 0.1, 2.6, 1.3, step=0.1)

st.write("")
if st.button("Predict type of Iris"):
    features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    result = predict(features)
    st.success(f"Predicted species: {result[0]}")

st.write("")
