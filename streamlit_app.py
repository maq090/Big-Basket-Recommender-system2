import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Big Basket Product Recommendation System",page_icon="logo.png",layout="wide")

data=pd.read_csv('preprocessed_with_cluster_label.csv')
st.title('Big Basket Product Recommendation System')
st.write(data)
