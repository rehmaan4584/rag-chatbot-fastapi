from google.genai import types

class ChatService():
    def __init__(self, client, tools):
        self.client = client
        self.tools = tools
        self.chat_history = {}

    def stream_reply(self, session_id: str, message: str):
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        history = self.chat_history[session_id]
        history.append({"role": "user", "content": message})

        gemini_format = [
            {"role": h["role"], "parts": [{"text": h["content"]}]}
            for h in history
        ]
        full_reply = "" 
        stream = self.client.models.generate_content_stream(
            model="gemini-3.8-flash",
            contents=gemini_format,  
            config=types.GenerateContentConfig(
                tools = self.tools
            )
        )
        for chunk in stream:
            if chunk.text:
                full_reply += chunk.text   
                yield chunk.text           
        history.append({"role": "model", "content": full_reply})