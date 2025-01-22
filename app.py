import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle

# Load the FAISS index
def load_faiss_index(index_path="faiss_index.pkl"):
    with open(index_path, "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

# Streamlit app
st.title("Document Search ")
st.write("Enter a query to search the documents.")

# Load FAISS index
vector_store = load_faiss_index()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Input box for query
query = st.text_input("Enter your query:")

# Search and display results
if query:
    # Pass raw query text to FAISS
    results = vector_store.similarity_search(query, k=1)

    st.write("### Search Results:")
    for i, result in enumerate(results):
        st.write(f"**Result {i+1}:** {result.page_content}")

# import streamlit as st
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# import pickle

# def load_vector_store(path="vector_store.pkl"):
#     with open(path, "rb") as f:
#         vector_store = pickle.load(f)
#     return vector_store

# # Streamlit app
# st.title("Document Search with FAISS")
# st.write("Enter a query to search the documents.")

# # Load FAISS index
# vector_store = load_vector_store()

# # Input box for query
# query = st.text_input("Enter your query:")

# # Search and display results
# if query:
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
#     results = retriever.get_relevant_documents(query)

#     st.write("### Search Results:")
#     for i, result in enumerate(results):
#         st.write(f"**Result {i+1}:** {result.page_content}")

