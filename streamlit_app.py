#references: streamlit.io.docs
#https://github.com/vermaayush680/
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
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
# used gzip form 
#https://stackoverflow.com/questions/24906126/how-to-unpack-pkl-file
@st.cache(ttl=168*3600)
def loadglove(name):
    """ function to laod glove vector file")"""
    with gzip.open(name,'rb') as f:
        model=pickle.load(f)
        return model

file='glove_vectors.pkl.gz'                  
model=loadglove(file)
glove_words=set(model.keys())

# Featurization of categorical and text features
@st.cache
def ohe(feature):
    """to one hot encoding of categorical feature"""
    vectorizer=CountVectorizer()
    return vectorizer.fit_transform(feature)

#one hot encoding categorical features
category_ohe=ohe(data['category'].values)

sub_category_ohe=ohe(data['sub_category'].values)

brand_ohe=ohe(data['brand'].values)

type_ohe=ohe(data['type'].values)

@st.cache(ttl=168*3600)
def tfidf_w2v(feature):
    """to get tfidf weighted w2v for product featue(text data) and returns a list of tfidf-w2v for product title"""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_product = tfidf_vectorizer.fit_transform(feature)
    
    # we are converting a dictionary with word as a key, and the idf as a value
    dictionary = dict(zip(tfidf_vectorizer.get_feature_names(), list(tfidf_vectorizer.idf_)))
    tfidf_words = set(tfidf_vectorizer.get_feature_names())
    
    product_tfidf_w2v_vectors = []; # the avg-w2v for each product is stored in this list
    for product in (feature): # for each product title
        vector = np.zeros(300) # as word vectors are of zero length
        tf_idf_weight =0; # num of words with a valid vector in the product
        for word in product.split(): # for each word in product title
            if (word in glove_words) and (word in tfidf_words):
                vec = model[word] # getting the vector for each word
                # here we are multiplying idf value(dictionary[word]) and the tf value((product.count(word)/len(product.split())))
                tf_idf = dictionary[word]*(product.count(word)/len(product.split())) # getting the tfidf value for each word
                vector += (vec * tf_idf) # calculating tfidf weighted w2v
                tf_idf_weight += tf_idf
        if tf_idf_weight != 0:
            vector /= tf_idf_weight
        product_tfidf_w2v_vectors.append(vector)
    return product_tfidf_w2v_vectors

product_tfidf_w2v_vectors=tfidf_w2v(data['product'])

#stacking all vectorized features for computing cosine similarity
X_tfidf_w2v = hstack ((product_tfidf_w2v_vectors,category_ohe,sub_category_ohe,brand_ohe,type_ohe,data['sale_price'].values.reshape(-1,1), \
             data['negative'].values.reshape(-1,1),data['neutral'].values.reshape(-1,1),data['positive'].values.reshape(-1,1), \
             data['compound'].values.reshape(-1,1),data['cluster_label'].values.reshape(-1,1))).tocsr()

#defining main function to get similar products

@st.cache
def get_similar(prod_index,num_results=11):
    # prod_index: product index in the given data
    # num_results: number of similar products to show
    
    # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    cosine_sim=cosine_similarity(X_tfidf_w2v,X_tfidf_w2v[prod_index])
    
    # np.argsort will return indices of the nearest products 
    indices = np.argsort(cosine_sim.flatten())[-num_results:-1]
    # -1 given to exclude the searched product itself from showing in recommendations as cosinine similarity will be 1 for same product
    # flipping the indices so that the product with more similarity is shown first
    # argsort will do sorting of indices from smallest to largest value
    indices=np.flip(indices)
    #psimilarity will store the similarity 
    psimilarity  = np.sort(cosine_sim.flatten())[-num_results:-1]
    psimilarity = np.flip(psimilarity)
    
    df=data[['product','discount_%']].loc[indices]
    df['discount_%']=df['discount_%']*0.5/100 # multiplied by 0.5 to give half weightage to discount % and divided by 100 to convert 
    # percentage to decimal
    df['similarity']=psimilarity.tolist() # adding similarity scores as s new column to df
    
    df['rank_score']= df['discount_%']+df['similarity'] # creating rank score by adding similarity and discount
    
    df=df.sort_values(by='rank_score',ascending=False)
    
    index=list(df.index) # getting indices of similar products to fetch them from main data
    
    return data.loc[index,['product','sale_price']] # returns product title and sale price of similar products

# button to select a product
selected_product=st.selectbox("Type or select product from dropdown",data['product'].values)

prod_index=int(data[data['product']==selected_product].index.values[0])
            
if st.button("Show Similar Products"):
    st.subheader("Similar Products for :" +data['product'].loc[prod_index])
    similar_products=get_similar(prod_index) # num_results argument not passed as it is given a default value in function
    st.dataframe(similar_products)


