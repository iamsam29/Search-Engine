import pandas as pd
import chromadb
import streamlit as st
import ast
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer


def preprocess_text(text):

    # Remove timestamps
    clean_text = re.sub(r'\b\d+\b\s*?\n?', '', text)
    clean_text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n?', '', clean_text)

    # Define the pattern to match website links
    pattern = r'https?://\S+|www\.\S+'
    # Replace website links with an empty string
    cleaned_text = re.sub(pattern, '', clean_text)

    # Remove special characters
    pattern = r'[^a-zA-Z0-9\s]'
    textt = re.sub(pattern, '', cleaned_text)

    # Lowercase conversion
    textt = textt.lower()

    # Tokenization
    tokens = word_tokenize(textt)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Join tokens back into text
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text

model = SentenceTransformer('all-MiniLM-L6-v2')
def encode_text(text):
  embedding = model.encode(text)
  return embedding.tolist()


new_df = pd.read_csv('D:/Codes_VS/New folder/bert_embedded_data_final.csv')
# Initialize PersistentClient with the desired path to save the database
client = chromadb.PersistentClient(path="D:/Codes_VS/New folder")

collection = client.get_or_create_collection(
    name="app_semantic_search_engine",  # Specify a name for the collection
    metadata={"hnsw:space": "cosine"}  # Specify metadata for the collection
)

# Assuming 'new_df' is your DataFrame containing the data
for index, row in new_df.iterrows():
    # Extract necessary columns including embeddings
    num = str(row['num'])  # Convert num to string
    name = row['name']
    
    # Parse the embeddings from string to list
    embeddings_str = row['bert_embeddings']
    embeddings = ast.literal_eval(embeddings_str)
    
    # Insert data into ChromaDB
    collection.add(
        ids=[num],  # Assuming 'num' is the ID
        embeddings=[embeddings],  # Pass embeddings in a list
        documents=[name]  # Adjust documents parameter
    )


def query_and_display_results(collection, query_text, n_results=10):
    # Preprocess the query text and encode it into a vector
    query_processed = preprocess_text(query_text)
    query_vector = encode_text(query_processed)

    # Query the collection
    query_results = collection.query(
        [query_vector],  # Pass the query vector as a list
        n_results=n_results)  # Retrieve top n_results

    # Sort the results by similarity score in descending order
    sorted_results = sorted(zip(query_results['documents'][0], query_results['distances'][0]), key=lambda x: x[1], reverse=True)

    # Create a DataFrame from the sorted results
    df = pd.DataFrame(sorted_results, columns=['Document', 'Similarity Score'])

    # Display the DataFrame in a table format
    st.table(df)

# Example usage:
query_text = st.text_input("Enter the query:")
if query_text:
    query_and_display_results(collection, query_text)