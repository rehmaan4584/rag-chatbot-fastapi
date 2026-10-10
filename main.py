import os
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from fastapi.responses import StreamingResponse
from schemas import ChatRequest
from tools import get_time, get_weather
from google.genai import types
from rag_service import ask_rag

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI()
chat_history = {}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/rag")
def rag(request: ChatRequest):
    ans = ask_rag(request.message)
    return {"reply": ans, "sessionId": request.session_id}

@app.post("/chat")
def chat(request: ChatRequest):
    if request.session_id not in chat_history:
        chat_history[request.session_id] = []
    history = chat_history[request.session_id]
    history.append({"role": "user", "content": request.message})

    gemini_format = [
        {"role": h["role"], "parts": [{"text": h["content"]}]}
        for h in history
    ]

    def generate():
        full_reply = ""  # poora jawab yahan jama karenge, history mein save karne ke liye
        stream = client.models.generate_content_stream(
            model="gemini-3.8-flash",
            contents=gemini_format,   # ab poori history ja rahi hai
            config=types.GenerateContentConfig(
                tools=[get_weather, get_time]
            )
        )
        for chunk in stream:
            if chunk.text:
                full_reply += chunk.text   # chunk ko jama karo
                yield chunk.text           # aur turant bhej bhi do

        # stream khatam hone ke baad, poora jawab history mein save karo
        history.append({"role": "model", "content": full_reply})

    return StreamingResponse(generate(), media_type="text/plain")
    
