import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("🌍 AI Travel Concierge - RAG Assistant")

api_key = st.text_input("Enter Groq API Key", type="password")

uploaded_file = st.file_uploader("Upload Travel Guide PDF", type="pdf")

if uploaded_file and api_key:

    with open("temp.pdf","wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=0.7
    )

    query = st.text_input("Ask a question from the document")

    if query:

        docs = retriever.invoke(query)

        context = "\n".join([d.page_content for d in docs])

        response = llm.invoke(
        f"""
        Use the context below to answer.

        Context:
        {context}

        Question:
        {query}
        """
        )

        st.subheader("🤖 AI Answer")
        st.write(response.content)
