import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import load_pdf, ask_question, save_message, get_chat_history

app = FastAPI(title="AI Generalist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    question: str
    domain: str
    session_id: str = None
    user_id: str = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list = []

@app.get("/")
def root():
    return {"status": "AI Generalist API is running"}

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    domain: str = Form(default="law")
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks_created = load_pdf(file_path, domain)
    return {
        "message": f"{file.filename} uploaded and processed successfully",
        "chunks_created": chunks_created
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())
    answer, sources = ask_question(request.question, request.domain)
    save_message(session_id, request.domain, request.question, answer, request.user_id, sources)

    return ChatResponse(answer=answer, session_id=session_id, sources=sources)

@app.get("/history/{session_id}")
async def history(session_id: str):
    messages = get_chat_history(session_id)
    return {"messages": messages}

@app.get("/conversations")
async def get_conversations(user_id: str = None):
    from supabase import create_client
    import os
    supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    
    query = supabase_client.table("conversations").select("*").order("updated_at", desc=True).limit(20)
    
    if user_id:
        query = query.eq("user_id", user_id)
    
    result = query.execute()
    return {"conversations": result.data}