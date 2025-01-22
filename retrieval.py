import pickle
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

def load_vector_store(path="faiss_index.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    llm = OpenAI(model="text-davinci-003")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

if __name__ == "__main__":
    vector_store = load_vector_store()
    qa_chain = create_qa_chain(vector_store)
    
    query = input("Enter your query: ")
    print("Answer:", qa_chain.run(query))

