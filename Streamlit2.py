import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import warnings
import nltk
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
import re
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn.preprocessing import LabelEncoder
from PIL import Image

st.set_page_config(page_title="Big Basket Product Recommendation System",page_icon="logo.png",layout="wide")

st.title('Big Basket Product Recommendation System')

# we use the list of stop words that are downloaded from nltk lib.
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# computing sentiment score for description feature
nltk.download('vader_lexicon')


df=pd.read_csv('train_preprocessed_with_clusterlabels.csv') # loading preprocessed data


#loading the glove_vector file which is in zipped form
# used gzip form 
#https://stackoverflow.com/questions/24906126/how-to-unpack-pkl-file

with gzip.open('glove_vectors.pkl.gz' ,'rb') as f:
    model=pickle.load(f)
    glove_words=set(model.keys())

# tfidf vectorizer
tfidf=TfidfVectorizer()
tfidf_description=tfidf.fit_transform(df['description'])

# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
tfidf_words = set(tfidf.get_feature_names())


# function to label encode categorical columns
def label_encode_columns(df, columns, encoders=None):
    if encoders is None:
        encoders = {}
    
        for col in columns:
            unique_values = list(df[col].unique())
            unique_values.append('Unseen')
            le = LabelEncoder().fit(unique_values)
            df[col] = le.transform(df[[col]])
            encoders[col] = le
    
    else:
        for col in columns:
            le = encoders.get(col)
            df[col] = [x if x in le.classes_ else 'Unseen' for x in df[col]]
            df[col] = le.transform(df[[col]])

    return df, encoders

columns=['category', 'sub_category', 'brand', 'type']
df, le_for_test=label_encode_columns(df, columns)

@st.cache
def loadpickle(filename):
    """to load pickle file"""
    with open (filename,'rb') as f:
        data_pickle=pickle.load(f)
        
        return data_pickle
    
# loading X_train, hstack of all vectors to compute cosine similarity with query product
X_train=loadpickle('X_train.pkl')

def missing_features(data):
    """to check any missing or blank or nan values in qury product """
    data.replace('None',np.nan,inplace=True) # replace None with Nan
    data.replace(r'^\s*$',np.nan,regex=True,inplace=True) # replace empty string with NaN
    
   # first to check whether the data has the 8 listed columns[category,sub_category,brand,type,description,sale_price,market_price]
   # to check for any missing values
    if(data.shape[1]<7) or data.isna().any().any():
        st.write('Please check datapoint\n')
        st.write('The data has missing values in'+ str(list(data.columns[data.isna().any()]))+ 'columns')
        
        return False
    else:
        return True
    
# function to check for sale price range 
sale_price_minima=df['sale_price'].min()-(df['sale_price'].min()*0.15) # 15 less than min 
sale_price_maxima=df['sale_price'].max()+(df['sale_price'].max()*0.15)

def salepricecheck(data,train_data):
    '''function to check data has sale price relavant to other produts in same brand or whole train data'''
    if data['sale_price'].values.size!=0: #https://stackoverflow.com/questions/11295609/how-can-i-check-whether-a-numpy-array-is-empty-or-not
        if data['brand'].notna().all():
            g=train_data.groupby(['brand'])['sale_price'] # grouping train data based on brand to get group of query product brand
            minimum=g.get_group((data['brand'].values[0])).min() # getting minimum sale price of query product brand in train data
            minimum = minimum-(minimum*0.15) # 15% tolerance
            
            maximum=g.get_group((data['brand'].values[0])).max()
            maximum = maximum + (maximum*0.15)
            
            if minimum <= float(data['sale_price'].values[0]) <= maximum:
                return True
            else:
                st.write('Warning: The sale_price of query product is not in range of other products in same brand,Check sale_price')
                return False
        else:
            if sale_price_minima <= float(data['sale_price'].values[0]) <= sale_price_maxima: # if brand is not available then will see in whole             #train data
                return True
            else:
                st.write('Warning:The sale_price of query product is not in range of train data products,please check sale_price')
                return False
    else:
        st.write('Error:No sale_price for query product given')    
        return False
    
 

from nltk.sentiment.vader import SentimentIntensityAnalyzer
def get_scores(data):
    """retuens sentiment analysis scores for description feature"""
    sia=SentimentIntensityAnalyzer()
    
    negative=[]
    neu=[]
    pos=[]
    compound=[]
    if 'description' in data.columns:
        for value in (data['description']):
            i=sia.polarity_scores(value)['neg']
            j=sia.polarity_scores(value)['neu']
            k=sia.polarity_scores(value)['pos']
            l=sia.polarity_scores(value)['compound']
            
            negative.append(i)
            neu.append(j)
            pos.append(k)
            compound.append(l)
            
    data['negative']=negative
    data['neutral']=neu
    data['positive']=pos
    data['compound']=compound
    
    return data

# function to get tfidf weighted word2vec

def get_tfidf_w2v(data):
    """retuens tfidf weighted w2v for description feature"""
    
    description_tfidf_w2v_vectors = []; # the avg-w2v for each description is stored in this list
    if 'description' in data.columns:
        for description in (data['description']):
            vector = np.zeros(300) # as word vectors are of zero length
            tf_idf_weight =0; # num of words with a valid vector in the description
            for word in description.split(): # for each word in description
                if (word in glove_words) and (word in tfidf_words):
                    vec = model[word] # getting the vector for each word
                    # here we are multiplying idf value(dictionary[word]) and the tf value((description.count(word)/len(product.split())))
                    tf_idf = dictionary[word]*(description.count(word)/len(description.split())) # getting the tfidf value for each word
                    vector += (vec * tf_idf) # calculating tfidf weighted w2v
                    tf_idf_weight += tf_idf
            if tf_idf_weight != 0:
                vector /= tf_idf_weight
            description_tfidf_w2v_vectors.append(vector)
    
    return description_tfidf_w2v_vectors

def categorical_preprocess(text):
    """to preprocess categorical features,use .apply for applying function"""
    text=text.str.replace('&','_') # replacing & with _
    text=text.str.replace(',','_') # replacing , with _
    text=text.str.replace("'",'') #replacing '' with ''(no space)
    text=text.str.replace(" ",'') # removing white spaces
    text=text.str.lower() # to lower case
    text=text.str.strip() # removing trailing and leading white space
    
    return text

# function to preprocess description text feature

def preprocess_description(text):
    """ Function which does preprocesiing on prodcut title feature,
        removes stopwords, replaces special character with space, converts to lower case,
    """
    preprocessed_description=[]
    for description in text:
        
        #Delete all the data which are present in the brackets
        description = re.sub(r'\([^()]*\)',' ',description)
        
        #removing urls
        description = re.sub(r'http\S+',' ',description)
        description = re.sub('[^A-Za-z]+', ' ', description) # remove all characters except a-z and A-Z and replace with white space
        # https://gist.github.com/sebleier/554280
        description = ' '.join(word for word in description.split() if word.lower() not in stop_words) # removing stop words
        description = ' '.join(word for word in description.split() if len(word)>2) # removing single letter and two letter words
        description = description.lower().strip()
        preprocessed_description.append(description)
        
    return preprocessed_description

cluster_centers=loadpickle('cluster_centers.pkl') # loading cluster centre of train data for assigning to the query datapoint

def get_clusterlabel(X,means=cluster_centers):
    """ to get cluster label"""
    minimum=sys.maxsize # initializing minimum as maximum integer so that the distances will be less than that
    index=-1
    for i in range(len(means)):
        dis=np.linalg.norm(X - means[i]) #https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
        
        if (dis<minimum):
            minmum=dis
            index=i
            
    return index

def get_similar_products(query,train_data=df,X_train=X_train,num_results=11):
    """function to give similar products from train_data for query product """
    
    # query: query product
    # train_data: preprocessed train data with all features
    #X_train: matrix to compute cosine similarity
    # num_results: number of similar products to show
    
    if missing_features(query):
        
        # preprocessing categorical columns
        query[['category','sub_category','brand','type']]=query[['category','sub_category','brand','type']].apply(categorical_preprocess)
        
        # encoding categorical features category,sub_category,brand,type
        columns=['category', 'sub_category', 'brand', 'type']
        query, encoders1=label_encode_columns(query, columns, le_for_test) # using encoders =encoders got by fitting on
        
        if salepricecheck(query,train_data):
            # preprocessing description
            query['description']=preprocess_description(query['description'].values)
        
            #calculating discount_%
            if 'discount_%' not in query.columns:
                query['discount_%']=(query['market_price']-query['sale_price'])/query['market_price']
        
            query=get_scores(query) # to get sentiment scores
        
            # scaling sale price
            scaler = MinMaxScaler()
            scaler.fit(train_data['sale_price'].values.reshape(-1,1))
            query['sale_price_scaled']=scaler.transform(query['sale_price'].values.reshape(-1,1))
        
            # to get cluster label
            X_q=np.hstack((query['sale_price_scaled'].values.reshape(-1,1),query['discount_%'].values.reshape(-1,1), \
                           query['negative'].values.reshape(-1,1),query['neutral'].values.reshape(-1,1), \
                           query['positive'].values.reshape(-1,1),query['compound'].values.reshape(-1,1)))
        
            query['cluster_label']=get_clusterlabel(X_q) # function to classify item to nearest means(cluster_centres)
        
            tfidf_w2v_vector=get_tfidf_w2v(query) # function to vectorize description text after preprocessing
        
            
            # stacking all values
            #https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html#numpy.concatenate
            X=np.hstack((tfidf_w2v_vector,query['category'].values.reshape(-1,1),query['sub_category'].values.reshape(-1,1), \
                         query['brand'].values.reshape(-1,1),query['type'].values.reshape(-1,1), \
                         query['sale_price_scaled'].values.reshape(-1,1),query['discount_%'].values.reshape(-1,1), \
                         query['negative'].values.reshape(-1,1),query['neutral'].values.reshape(-1,1), \
                         query['positive'].values.reshape(-1,1),query['compound'].values.reshape(-1,1), \
                         query['cluster_label'].values.reshape(-1,1)))
        
            # till now we have preprocessed and vectorized query product
            # now will compute cosine similarities and suggest similar products based on cosine similarity
            cosine_sim=cosine_similarity(X_train,X)
            # np.argsort will return indices of the nearest products 
            indices = np.argsort(cosine_sim.flatten())[-num_results:-1]
            # -1 given to exclude the searched product itself from showing in recommendations as cosinine similarity will be 1 for same product
            # flipping the indices so that the product with more similarity is shown first
            # argsort will do sorting of indices from smallest to largest value
            indices=np.flip(indices)
            ##psimilarity will store the similarity 
            psimilarity  = np.sort(cosine_sim.flatten())[-num_results:-1]
            psimilarity = np.flip(psimilarity)
        
            
            st.markdown('\nTop '+str(num_results-1)+' Similar products for "'**query['product'].values[0]**'" are:')
            
            data=train_data[['product','discount_%']].loc[indices]
            data['similarity']=psimilarity.tolist() # adding similarity scores as a new column to data
           
            index=list(data.index) # getting indices of similar products to fetch them from main data
            
            return train_data.loc[index,['product']]
    else:
        print('Warning:Please check query point for any missing or incomplete information')


#image=Image.open('demo.png') # sample data point image 
# file uploader to upload query datapoint, single point at a time
image=Image.open('demo.png')
st.image(image,caption="sample data point to upload")
uploaded_file=st.file_uploader('Upload csv file of single query product at a time with all columns')
#st.image(image, caption='Sample datapoint')

num_results=11
if uploaded_file is not None:
    query=pd.read_csv(uploaded_file)
    if missing_features(query):
        if st.button("Explore more for Similar Products"):
            st.subheader('The queried product is:'+query['product'].values[0])
            similar_products=get_similar_products(query)
            st.dataframe(similar_products)
            st.caption('Clear the upload and input new datapoint to check for new product')
        
    
