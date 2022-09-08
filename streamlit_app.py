#references: streamlit.io.docs
#https://github.com/vermaayush680/
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from scipy.sparse import hstack
import pickle
from scipy.sparse import hstack

st.set_page_config(page_title="Big Basket Product Recommendation System",page_icon="logo.png",layout="wide")

st.title('Big Basket Product Recommendation System')

@st.cache
def load_data(filename):
    """ function to load data"""
    data=pd.read_csv(filename)
    return data
  
file='preprocessed_with_clusterlabel.csv'
data=load_data(file)

#loading the glove_vector file which is in zipped form

glovefile='glove_vectors.7z'
