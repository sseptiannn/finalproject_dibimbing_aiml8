import streamlit as st
import pandas as pd
# from src.preprocessing import DataPreprocessor
# from src.clustering import CustomerClustering


st.title("Customer Segmentation & Default Risk Analysis")


uploaded_file = st.file_uploader("Upload customer data CSV")


# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     pre = DataPreprocessor()
#     df = pre.feature_engineering(df)


#     features = ['payment_ratio', 'late_payment_frequency']
#     X = pre.scale_features(df, features)


#     cluster_model = CustomerClustering(n_clusters=3)
#     df['cluster'] = cluster_model.fit_predict(X)


# st.write("### Clustered Customers Test")
# st.dataframe(df)