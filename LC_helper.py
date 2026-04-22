import os
from pathlib import Path
from dotenv import load_dotenv
_ = load_dotenv(Path(__file__).parent / ".env")  # Load environment variables from .env file

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain_groq import ChatGroq
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_classic.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.question_answering.chain import load_qa_chain




persist_directory = "rag_chatBot/chroma_db"

question_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Answer the question based on the context:\n{context}\nQuestion:{question}"
    )
refine_prompt = PromptTemplate(input_variables=["existing_answer","context", "question"], template=
                               """We have existing answer: {existing_answer}
                               Improve it using new context:
                               {context}
                               Question: {question}
                               Refined answer:"""
                               )

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    api_key=os.environ["GROQ_API_KEY"]
)

condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the conversation and a follow-up question,
rephrase it as a standalone question.

Chat History:
{chat_history}

Question:
{question}

Standalone question:"""
)

question_generator = LLMChain(
    llm=llm,
    prompt=condense_prompt
)

doc_chain = load_qa_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=question_prompt,
    refine_prompt=refine_prompt,
    document_variable_name = "context"
)

def pdf_chain(file, k=2):


    # Load documents
    loader = PyPDFLoader(file)
    documents = loader.load()

    # Split documents into chunks
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_spliter.split_documents(documents=documents)

    # define embedding
    embedding = GPT4AllEmbeddings(model="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={"allow_download": "true"})

    # Create vector database
    db = Chroma.from_documents(docs, embedding)
    # Create retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    #memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the Conversational Retrieval Chain
    qa = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        memory=memory
    )
    return qa
