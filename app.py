
import streamlit as st
from LC_helper import pdf_chain
import tempfile

st.title("RAG CHATBOT")

file = st.file_uploader("Upload a PDF file", type=".pdf")

if file is not None:
    # Save uploded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(file.read())
        temp_file_path = tmpfile.name
    qa_chain = pdf_chain(temp_file_path)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.chat_input("Ask a question about the PDF:")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = qa_chain({"question": user_input, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
        
        # Display the conversation
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**👤 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {message['content']}")