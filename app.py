import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ------------ CONFIG ------------

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ------------ PDF & CHUNKING ------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    return chunks


# ------------ VECTOR STORE (HUGGINGFACE) ------------

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# ------------ RAG CHAIN (GEMINI LLM) ------------
from langchain_google_genai import ChatGoogleGenerativeAI

def get_conversational_chain():
    prompt_template = """
You are a helpful AI assistant. Use the following context to answer the question at the end.

Context:
{context}

Question: {question}

Answer in clear, simple terms for a college student.
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # from your list_models output
        temperature=0.3,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    def combine_docs(docs):
        return "\n\n".join(docs)

    rag_chain = (
        {"context": combine_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain



# ------------ USER QUESTION HANDLER ------------

def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("Please upload PDFs and click 'Process' first.")
        return

    embeddings = get_embeddings()
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    docs = new_db.similarity_search(user_question, k=3)

    # Convert to pure text list (handles both Document and str cases)
    texts = [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in docs
    ]

    chain = get_conversational_chain()
    answer = chain.invoke({"context": texts, "question": user_question})

    print(answer)
    st.write(answer)


# ------------ STREAMLIT UI ------------

def main():
    st.title("ChatBot with PDF Documents")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF documents here and click on 'Process'",
            accept_multiple_files=True,
        )
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete!")


if __name__ == "__main__":
    main()
