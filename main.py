import os
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from fastapi.responses import StreamingResponse
from schemas import ChatRequest
from tools import get_time, get_weather
from rag_service import RagService
from chat_service import ChatService

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
rag_service = RagService(client)
rag_service.setup_rag()   
def search_policy(question: str) -> str:
    """Searches the company's leave and work-from-home policy documents and returns the most relevant passage for the question."""
    return rag_service.retrieve(question)
tools = [get_time, get_weather, search_policy]
chat_service = ChatService(client, tools)

app = FastAPI()
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/rag")
def rag(request: ChatRequest):
    ans = rag_service.ask_rag(request.message)
    return {"reply": ans, "sessionId": request.session_id}

@app.post("/chat")
def chat(request: ChatRequest):
    return StreamingResponse( chat_service.stream_reply(request.session_id,request.message),
    media_type="text/plain" )
    
