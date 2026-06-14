import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from supabase import create_client
from upstash_redis import Redis

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

def load_pdf(file_path: str, domain: str = "law"):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    filename = os.path.basename(file_path)

    # Save document record to Supabase
    doc_result = supabase.table("documents").insert({
        "filename": filename,
        "domain": domain
    }).execute()

    document_id = doc_result.data[0]["id"]

    # Embed each chunk and store in Supabase
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk.page_content)
        supabase.table("document_chunks").insert({
            "document_id": document_id,
            "chunk_text": chunk.page_content,
            "embedding": embedding
        }).execute()

    return len(chunks)

def get_relevant_chunks(question: str, domain: str, k: int = 15):
    question_embedding = embeddings.embed_query(question)
        
    result = supabase.rpc("match_chunks", {
        "query_embedding": question_embedding,
        "match_count": k,
        "filter_domain": None if domain == "general" else domain
    }).execute()

    return [row["chunk_text"] for row in result.data]

def ask_question(question: str, domain: str) -> str:
    cache_key = f"{domain}:{question.strip().lower()}"
    cached = redis.get(cache_key)
    if cached:
        return cached

    chunks = get_relevant_chunks(question, domain)

    if not chunks:
        return "I could not find relevant information in the available documents for this query."

    context = "\n\n".join(chunks)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an expert {domain} consultant in Pakistan.
    Answer the question based only on the following context from official documents.
    If you cannot find the exact answer, provide the most relevant information from the context that relates to the question.

    Context: {context}

    Question: {question}
    """)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "context": context,
        "question": question,
        "domain": domain
    })

    if "could not find" not in answer.lower():
        redis.set(cache_key, answer)

    return answer

def save_message(session_id: str, domain: str, user_message: str, ai_response: str, user_id: str = None):
    existing = supabase.table("conversations").select("*").eq("session_id", session_id).execute()
    
    if existing.data:
        messages = existing.data[0]["messages"]
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "ai", "content": ai_response})
        supabase.table("conversations").update({
            "messages": messages,
            "updated_at": "now()"
        }).eq("session_id", session_id).execute()
    else:
        supabase.table("conversations").insert({
            "session_id": session_id,
            "domain": domain,
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "ai", "content": ai_response}
            ]
        }).execute()

def get_chat_history(session_id: str):
    result = supabase.table("conversations").select("*").eq("session_id", session_id).execute()
    if result.data:
        return result.data[0]["messages"]
    return []