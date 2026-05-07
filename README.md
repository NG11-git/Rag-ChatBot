# RAG CHATBOT
A powerful **Retrieval-Augmented Generation(RAG)** system designed to provide context-aware answers by querying your own documents. This project combines efficient document retrieval with advanced language modeling to minimize hallucination and provide accurate response.

## Features
* **Contextual Retrieval** : Efficiently searches through [PDF/Docs/Webpage] using vesctor embeddings. (This one is build for PDF)

*  **LLM Integration** : Powered by [e.g., OpenAI GPT-4/Anthropic/Llama 3] for natural and accurate response. (The model is Llama 3 from Groq)

*  **Semantic Search** : Uses [e.g., Pinecone/FAISS/ChromaDB/Weaviate] to find information based on meaning, not just keywords. (For vector storage used chromadb)

*  **Streamlined UI** : A clean interface built with Streamlit for easy interaction.


## Tech Stack

*Language* : Python
*Framework* : LangChain
*Embeddings* : GPT4AllEmbeddings
*Vector DB* : ChromaDB
*Frontend* : Streamlit

## Getting Started
**Prerequisites**

* Python 3.9+
* An API Key for [LLm Provider]

### Installation

1. Clone the repository:

```Bash
git clone https://github.com/your-username/Rag-ChatBot.git
cd Rag-ChatBot
```

2. Set up a virtual environment:

```Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```Bash
pip install -r requirements.txt
```

4. Environment Variables:
Create a ```.env``` file in the root direcotry and add your keys:

``` Code snippet
API_KEY=your_key_here
DATABASE_URL=your_db_link
```

## Usage 
Run the application using the following command:
```Bash
streamlit run app.py
```
