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
    # Reduce chunk size to ensure smaller input to the model
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def truncate_context(context, max_tokens):
    """Truncate the context to fit within the token limit."""
    while len(context.split()) > max_tokens:
        context = context.rsplit(" ", 10)[0]  # Remove the last 10 words iteratively
    return context

def get_conversational_chain():
    prompt_template = """
    The provided context may be in any language, but you must always answer in English.
    Read the context carefully and answer the question as accurately and thoroughly as possible.
    If the answer is not present in the provided context, respond with:
    "Answer is not available in the context." Do not make up an answer.
    
    Context (may be in any language):\n{context}\n
    Question:\n{question}\n

    Answer (always in English):
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question, k=5)  # Limit the number of retrieved documents
    context = " ".join([doc.page_content for doc in docs])
    context = truncate_context(context, max_tokens=30000)  # Ensure the context fits within the token limit
    
    chain = get_conversational_chain()
    
    try:
        response = chain({"input_documents": [context], "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error: {str(e)}. Please try with a shorter question or fewer documents.")

    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")
    st.header("Chat with Multiple PDFs using Gemini")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed!")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
