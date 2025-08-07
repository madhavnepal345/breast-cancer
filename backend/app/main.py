from fastapi import FastAPI, Request
from pydantic import BaseModel
from qa_model import QAModel
from vector_db import VectorDatabase
from conversation import ConversationManager
from typing import Dict
import os
from fastapi.staticfiles import StaticFiles


app = FastAPI()
qa_model = QAModel()
vector_db = VectorDatabase()
conv_manager = ConversationManager()


if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")



class ChatRequest(BaseModel):
    query: str
    session_id: str = None
    enable_memory: bool = True



@app.get("/")
async def welcome():
    return {"status": "ok", "message": "FastAPI is running"}

@app.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    session = conv_manager.get_session(chat_request.session_id)
    
    # Resolve references using conversation history
    resolved_query = session.resolve_references(chat_request.query)
    
    # Retrieve relevant context
    search_results = vector_db.hybrid_search(resolved_query)
    contexts = [r["text"] for r in search_results[:3]]
    
    # Generate answer
    conversation_history = session.get_formatted_history() if chat_request.enable_memory else ""
    qa_response = qa_model.answer(
        context="\n\n".join(contexts),
        question=resolved_query,
        conversation_history=conversation_history
    )
    
    # Update conversation
    session.add_message("user", chat_request.query)
    if qa_response["answer"]:
        session.add_message("assistant", qa_response["answer"])
        # print(f"Answer: {qa_response['answer']}")
    
    return {
        "session_id": session.session_id,
        # "answer": qa_response["answer"] or "I'm not confident enough to answer that",
        "confidence": qa_response["confidence"],
        "sources": [r["metadata"] for r in search_results[:3]],
        "used_context": qa_response["context_used"]
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    return conv_manager.get_session(session_id).dict()



if __name__ == "__main__":
    import uvicorn

    # Directly run the ASGI app so `python main.py` starts the server.
    uvicorn.run(
        app,                   # the FastAPI instance
        host="0.0.0.0",        # change to "127.0.0.1" if you don't want external access
        port=8000,
        
    )