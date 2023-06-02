import streamlit as st

import pandas as pd

from dotenv import load_dotenv

import openai

import os

from openai.embeddings_utils import get_embedding, cosine_similarity

import pickle

EMBEDDING_MODEL = "text-embedding-ada-002"

# set path to embedding cache
embedding_cache_path = "recommendations_embeddings_cache.pkl"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("OpenAI Semantic Search")

st.write("This is a demo of OpenAI's Semantic Search API")

st.write("Upload a CSV file with a column of text you would like to search")

uploaded_file = st.file_uploader("Upload file", type=['csv'])

if uploaded_file is not None:

    if '.csv' in uploaded_file.name:
        df = pd.read_csv(uploaded_file, nrows=1000)
    
    elif '.xlsx' in uploaded_file.name:
        df = pd.read_excel(uploaded_file, nrows=1000)

    st.dataframe(df)

    id_column = st.selectbox('Which column would you like to search?', tuple(df.columns))

    text_input = st.text_input(
        "Enter some text 👇",
    )

    if text_input: 

        embedding = embedding_from_string(text_input)

        text_embeddings = [embedding_from_string(str(i)) for i in list(df[id_column])]

        scores = [cosine_similarity(i, embedding) for i in text_embeddings]

        df['scores'] = scores

        st.dataframe(df.sort_values('scores', ascending=False).head(10))

