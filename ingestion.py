from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import os
import pandas as pd
import pickle

# Initialize the embedding model with LangChain wrapper
def initialize_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_documents(data_path):
    """
    Reads and processes specific columns from CSV files in the data folder.
    """
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):  # Only process CSV files
            file_path = os.path.join(data_path, filename)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                print(f"Processing file: {filename}")
                print(f"Columns in file: {df.columns.tolist()}")  # Debug: Show column names

                # Process specific columns based on the file's content
                if 'Article' in df.columns:  # For Articles.csv
                    documents.extend(df['Article'].dropna().tolist())
                elif 'summaries' in df.columns:  # For books_summary.csv
                    documents.extend(df['summaries'].dropna().tolist())
                else:
                    print(f"Warning: No relevant text column found in {filename}.")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return documents

def store_embeddings_faiss(documents, output_path="faiss_index.pkl"):
    """
    Embeds the provided documents and stores them in a FAISS index.
    """
    if not documents:
        print("No documents to process.")
        return

    print("Generating embeddings...")
    embedding_model = initialize_embedding_model()  # Use LangChain wrapper
    doc_objects = [Document(page_content=doc) for doc in documents]

    print("Storing embeddings in FAISS...")
    vector_store = FAISS.from_documents(doc_objects, embedding_model)

    # Save FAISS index to file
    with open(output_path, "wb") as f:
        pickle.dump(vector_store, f)

    print(f"Embeddings successfully stored in {output_path}!")

if __name__ == "__main__":
    # Specify the data folder path
    data_path = "C:/Users/LENOVO/OneDrive/Desktop/RAG Pipeline/data"  # Update this path if needed
    # Ingest documents from the data folder
    documents = ingest_documents(data_path)
    # Store embeddings in the FAISS index
    if documents:
        store_embeddings_faiss(documents)
    else:
        print("No documents found for embedding.")






# from sentence_transformers import SentenceTransformer
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.docstore.document import Document
# import os
# import pickle

# # Load embedding model
# embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# def ingest_documents(data_path, output_path="faiss_index.pkl"):
#     documents = []
#     for file in os.listdir(data_path):
#         if file.endswith(".txt"):
#             with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
#                 content = f.read().strip()
#                 if content:  # Ensure the file has content
#                     documents.append(Document(page_content=content))
    
#     # Debugging: Print the number of documents
#     print(f"Number of documents processed: {len(documents)}")
#     if not documents:
#         print("No valid documents found. Please check the data folder.")
#         return

#     # Generate embeddings
#     print("Generating embeddings...")
#     embeddings = embedder.embed_documents([doc.page_content for doc in documents])

#     # Debugging: Check embeddings
#     print(f"Number of embeddings generated: {len(embeddings)}")
#     if not embeddings:
#         print("Failed to generate embeddings. Ensure documents have content.")
#         return

#     # Create FAISS vector store
#     vector_store = FAISS.from_documents(documents, embedder)
    
#     # Save vector store
#     with open(output_path, "wb") as f:
#         pickle.dump(vector_store, f)
#     print(f"Vector store saved to {output_path}")

# if __name__ == "__main__":
#     data_path = "C:/Users/LENOVO/OneDrive/Desktop/RAG Pipeline/data"
#     ingest_documents(data_path)