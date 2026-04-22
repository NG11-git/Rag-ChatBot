import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings


DATA_PATH = "rag_chatBot/Decision Tree.pdf"

def load_and_split_pdf(chunk_size=1000, chunk_overlap=200):
    # Load the PDF document
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Split the documents into chunks
    chunk = text_splitter.split_documents(documents)
    return chunk


Chroma_Path = "chroma_db"

gpt4_embedding = GPT4AllEmbeddings(model="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={"allow_download": "true"})

def create_vector_db(chunks, collection_name="pdf_chunks"):
    # Create a Vector DB 
    if os.path.exists(Chroma_Path):
        shutil.rmtree(Chroma_Path) # Clear the previous db and create new one
    
    vector_db = Chroma.from_documents(
        chunks, gpt4_embedding, collection_name=collection_name,
        persist_directory=Chroma_Path
        )
    print(f"Saved {len(chunks)}. chunks to {Chroma_Path}.")

def create_vector_db_from_pdf():
    # Create a vector database from a PDF file
    chunks = load_and_split_pdf()
    create_vector_db(chunks)
if __name__ == "__main__":
 create_vector_db_from_pdf()