import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = None

def load_pdf(file_path: str):
    global vectorstore
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)

    return len(chunks)

def ask_question(question: str, domain: str) -> str:
    if vectorstore is None:
        return "No documents uploaded yet. Please upload a PDF first."

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert {domain} consultant.
    Answer the question based only on the following context from the uploaded documents.
    If the answer is not in the context, say "I could not find this in the uploaded documents."

    Context: {context}

    Question: {question}
    """)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "domain": lambda _: domain
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)