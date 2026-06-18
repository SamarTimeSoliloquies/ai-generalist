import os
import uuid
from dotenv import load_dotenv
from llama_parse import LlamaParse
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
print("GROQ:", GROQ_API_KEY)
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown"
)
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
    documents = parser.load_data(file_path)
    
    full_text = "\n\n".join([doc.text for doc in documents])
    chunks = text_splitter.split_text(full_text)

    filename = os.path.basename(file_path)

    # Save document record to Supabase
    doc_result = supabase.table("documents").insert({
        "filename": filename,
        "domain": domain
    }).execute()

    document_id = doc_result.data[0]["id"]

    # Embed each chunk and store in Supabase
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk)
        supabase.table("document_chunks").insert({
            "document_id": document_id,
            "chunk_text": chunk,
            "embedding": embedding
        }).execute()

    return len(chunks)

def get_relevant_chunks(question: str, domain: str, k: int = 15):
    filter_domain = None if domain == "general" else domain
    question_embedding = embeddings.embed_query(question)

    vector_result = supabase.rpc("match_chunks", {
        "query_embedding": question_embedding,
        "match_count": k,
        "filter_domain": filter_domain
    }).execute()

    keyword_result = supabase.rpc("fts_match_chunks", {
        "query_text": question,
        "match_count": k,
        "filter_domain": filter_domain
    }).execute()

    scores = {}
    chunk_data = {}

    for rank, row in enumerate(vector_result.data):
        chunk_id = row["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
        chunk_data[chunk_id] = row

    for rank, row in enumerate(keyword_result.data):
        chunk_id = row["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
        chunk_data[chunk_id] = row

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:k]

    results = []

    for chunk_id in ranked_ids:
        row = chunk_data[chunk_id]

        results.append({
            "chunk": row["chunk_text"],
            "source": row.get("filename", "Unknown document")
        })

    return results


def rewrite_query(question: str, domain: str):
    """
    Rewrites a user's question into a retrieval-optimized query.
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an expert document retrieval assistant.

    Convert the user's question into a concise search query that would
    retrieve the most relevant sections from official Pakistani {domain}
    documents.

    Rules:
    - Preserve important keywords.
    - Expand abbreviations if helpful.
    - Use terminology likely to appear in legal, tax, accounting,
    medical, or regulatory documents.
    - Do NOT answer the question.
    - Do NOT explain.
    - Return ONLY the search query.

    Examples:

    Question: What is speculation business?
    Search Query: definition of speculation business income tax ordinance Pakistan

    Question: Who can contract a second marriage?
    Search Query: second marriage permission requirements family laws ordinance Pakistan

    Question: What is tax on dividends?
    Search Query: dividend tax rate income tax ordinance Pakistan

    Question:
    {question}
    """)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    try:
        rewritten_query = chain.invoke({
            "question": question,
            "domain": domain
        })

        return rewritten_query.strip()

    except Exception as e:
        print("QUERY REWRITE ERROR:", e)
        return question
    
    
def get_retrieved_context(question: str, domain: str):

    rewritten_query = rewrite_query(question, domain)

    print("\n==========================")
    print("Original Question:")
    print(question)
    print("\nRewritten Query:")
    print(rewritten_query)
    print("==========================\n")

    results_original = get_relevant_chunks(
        question,
        domain,
        k=8
    )

    results_rewritten = get_relevant_chunks(
        rewritten_query,
        domain,
        k=8
    )

    merged = []
    seen_chunks = set()

    for item in results_original + results_rewritten:
        chunk_text = item["chunk"]

        if chunk_text not in seen_chunks:
            seen_chunks.add(chunk_text)
            merged.append(item)

    merged = merged[:15]

    chunks = [item["chunk"] for item in merged]
    sources = [item["source"] for item in merged]

    return chunks, sources



def ask_question(question: str, domain: str):
    cache_key = f"{domain}:{question.strip().lower()}"

    cached = redis.get(cache_key)
    if cached:
        return cached, []

    chunks, sources = get_retrieved_context(
        question,
        domain
    )

    if not chunks:
        return (
            "I could not find relevant information in the available documents for this query.",
            []
        )

    context = "\n\n".join(chunks)

    unique_sources = []
    seen = set()

    for source in sources[:5]:
        if source not in seen:
            seen.add(source)
            unique_sources.append(source)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an expert {domain} consultant in Pakistan.

    Answer the question based only on the following context from official documents.

    If the answer is not available in the context,
    say that you could not find the information.

    Context:
    {context}

    Question:
    {question}
    """)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    try:
        answer = chain.invoke({
            "context": context,
            "question": question,
            "domain": domain
        })

    except Exception as e:
        print("GROQ ERROR:", e)
        return f"Error contacting LLM: {str(e)}", []

    if "could not find" not in answer.lower():
        redis.set(cache_key, answer)

    return answer, unique_sources

def save_message(session_id: str, domain: str, user_message: str, ai_response: str, user_id: str = None, sources: list = []):
    existing = supabase.table("conversations").select("*").eq("session_id", session_id).execute()
    
    if existing.data:
        messages = existing.data[0]["messages"]
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "ai", "content": ai_response, "sources": sources})
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
            {"role": "ai", "content": ai_response, "sources": sources}
        ]
    }).execute()

def get_chat_history(session_id: str):
    result = supabase.table("conversations").select("*").eq("session_id", session_id).execute()
    if result.data:
        return result.data[0]["messages"]
    return []