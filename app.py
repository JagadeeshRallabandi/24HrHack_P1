import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configure the Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
You are a helpful assistant capable of answering questions based on the provided PDF content. Your responses should be accurate, concise, and in fluent English.

Context:\n{context}\n
Question:\n{question}\n

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat with Multilingual PDFs", layout="wide")
    st.title("Chat with Multilingual PDFs")

    # Sidebar for PDF Upload
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed! You can now ask questions.")
            else:
                st.error("Please upload at least one PDF file.")

    # Initialize conversation history and current question/answer
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "current" not in st.session_state:
        st.session_state.current = {"question": "", "answer": ""}

    # User input for the question
    user_question = st.text_input("Ask a question about the uploaded PDFs:")
    if st.button("Submit Question"):
        if user_question:
            with st.spinner("Generating response..."):
                # Get response for the current question
                response = get_response(user_question)
                # Store the current question and answer temporarily
                st.session_state.current = {"question": user_question, "answer": response}
        else:
            st.error("Please enter a question.")

    # Display the current answer
    if st.session_state.current["question"]:
        st.markdown("### Current Question:")
        st.markdown(f"**Q:** {st.session_state.current['question']}")
        st.markdown(f"**A:** {st.session_state.current['answer']}")

    # Update the conversation history after asking a new question
    if user_question and st.session_state.current["question"]:
        st.session_state.conversation.append(st.session_state.current)
        st.session_state.current = {"question": "", "answer": ""}

    # Display conversation history
    if st.session_state.conversation:
        st.markdown("### Conversation History:")
        for idx, message in enumerate(st.session_state.conversation, 1):
            st.markdown(f"**Q{idx}:** {message['question']}")
            st.markdown(f"**A{idx}:** {message['answer']}")

if __name__ == "__main__":
    main()
