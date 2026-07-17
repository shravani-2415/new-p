import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

df = pd.read_csv('final_data.csv')


if not os.path.exists("similarities.pkl"):
    if st.button("Generate similarities"):
        cv = CountVectorizer(max_features=10000, stop_words = 'english')
        dtm = cv.fit_transform(df['tags'])
        dtm_df = pd.DataFrame(data = dtm.toarray(),  columns = cv.get_feature_names_out())
        similarities = cosine_similarity(dtm_df)
        pickle.dump(similarities, open('similarities.pkl', 'wb'))

names = sorted(df['title'].unique())

def get_movie_index(name):
    for i in df.index:
        if name == df.loc[i, 'title']:
            return i
    else:
        return -1

def get_movie_name(i):
    if i > len(df):
        return ""
    else:
        return df.loc[i, 'title']

name = st.selectbox("Select Movie You Watched", names)

if st.button("Recommend"):

    index = get_movie_index(name)
    if index == -1:
        st.error("Movie not found")
    else:
        similarities = pickle.load(open('similarities.pkl', 'rb'))
        st.write("Predicted next 5 movies:")
        similarity_index = similarities[index]
        similarity_index = list(enumerate(similarity_index))
        similarity_index = sorted(similarity_index, key = lambda x:x[1], reverse = True)
        print("Predicted next 5 movies")
        for i in range(1, 6):
            st.write(str(i) + ". " + get_movie_name(similarity_index[i][0]))