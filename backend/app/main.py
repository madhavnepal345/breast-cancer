from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from qa_model import QAModel
from vector_db import VectorDatabase
from conversation import ConversationManager
from typing import Optional
import os
from fastapi.staticfiles import StaticFiles
from config import settings
from fastapi.middleware.cors import CORSMiddleware
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Q&A System", version="1.0")

# Core modules
qa_model = QAModel()
vector_db = VectorDatabase()
conv_manager = ConversationManager()

# Enable CORS (adjust origins in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files setup
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ------------------------------
# Models
# ------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    enable_memory: bool = True
    min_confidence: float = 0.4  # Default confidence threshold


# ------------------------------
# Helpers
# ------------------------------
def is_medical_query(query: str) -> bool:
    """Check if the query is medical-related (regex word boundaries)."""
    medical_keywords = [
        'cancer', 'symptom', 'disease', 'illness', 'treatment', 'diagnosis',
        'medicine', 'drug', 'pain', 'health', 'medical', 'doctor', 'hospital',
        'condition', 'virus', 'infection', 'prescription', 'pharmacy', 'tumor',
        'therapy', 'clinical', 'patient', 'sick', 'ache', 'fever', 'cough'
    ]
    query_lower = query.lower()
    return any(re.search(rf"\b{kw}\b", query_lower) for kw in medical_keywords)


def get_enhanced_context(search_results, query):
    """Combine multiple contexts for better answers."""
    if not search_results:
        return None
    contexts = [result["text"] for result in search_results[:3]]
    return "\n\n".join(contexts)


# ------------------------------
# Routes
# ------------------------------
@app.get("/")
async def welcome():
    return {"status": "ok", "message": "FastAPI is running"}


@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # Ensure session exists
        if chat_request.session_id:
            session = conv_manager.get_session(chat_request.session_id)
        else:
            session = conv_manager.create_session()

        resolved_query = session.resolve_references(chat_request.query)

        # Adjust confidence threshold for medical queries
        is_medical = is_medical_query(resolved_query)
        adjusted_min_confidence = max(chat_request.min_confidence, 0.6) if is_medical else chat_request.min_confidence

        # Get relevant context - more for medical queries
        k = 5 if is_medical else 3
        search_results = vector_db.hybrid_search(resolved_query, k=k)

        if not search_results:
            return {
                "session_id": session.session_id,
                "answer": "I couldn't find any relevant information to answer your question.",
                "confidence": 0,
                "sources": [],
                "context_used": None,
                "is_medical": is_medical
            }

        # Use enhanced context (multiple sources)
        context = get_enhanced_context(search_results, resolved_query)

        # Confidence estimation
        confidence_estimate = qa_model.estimate_confidence(resolved_query, context)
        if confidence_estimate < adjusted_min_confidence:
            medical_disclaimer = (
                " Please consult a healthcare professional for accurate medical information."
                if is_medical else ""
            )
            return {
                "session_id": session.session_id,
                "answer": f"I don't have enough reliable information to answer this question accurately.{medical_disclaimer}",
                "confidence": confidence_estimate,
                "sources": [],
                "context_used": None,
                "is_medical": is_medical
            }

        # Generate answer
        qa_response = qa_model.answer(
            context=context,
            question=resolved_query,
            conversation_history=session.get_formatted_history() if chat_request.enable_memory else "",
            generate_full_answer=True
        )

        # Update conversation
        session.add_message("user", chat_request.query)

        if (qa_response.get("answer")
            and qa_response.get("confidence", 0) >= adjusted_min_confidence
            and not qa_response.get("error")):

            session.add_message("assistant", qa_response["answer"])

            # Prepare sources (top 2)
            sources = [result.get("metadata") for result in search_results[:2] if "metadata" in result]

            return {
                "session_id": session.session_id,
                "answer": qa_response["answer"],
                "confidence": qa_response["confidence"],
                "sources": sources,
                "context_used": qa_response.get("context_used", ""),
                "was_expanded": qa_response.get("was_expanded", False),
                "is_medical": is_medical
            }

        # Handle low confidence or error cases
        error_msg = qa_response.get("error", "")
        base_msg = (
            "I don't have enough reliable medical information to answer this question. Please consult a healthcare professional."
            if is_medical else
            "I'm not confident enough to provide a complete answer to that question. Could you please provide more details or rephrase your question?"
        )
        if error_msg:
            base_msg = f"{base_msg} (Error: {error_msg})"

        return {
            "session_id": session.session_id,
            "answer": base_msg,
            "confidence": qa_response.get("confidence", 0),
            "sources": [],
            "context_used": None,
            "is_medical": is_medical
        }

    except Exception as e:
        logger.error(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    try:
        session = conv_manager.get_session(session_id)
        return {
            "session_id": session.session_id,
            "messages": session.messages,
            "created_at": session.created_at,
            "updated_at": session.updated_at
        }
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        conv_manager.delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    return {
        "sessions": list(conv_manager.sessions.keys()),
        "count": len(conv_manager.sessions)
    }


@app.get("/status")
async def system_status():
    doc_count = vector_db.document_count()
    return {
        "document_loaded": doc_count > 0,
        "document_pages": doc_count,
        "active_sessions": len(conv_manager.sessions),
        "qa_model_loaded": hasattr(qa_model, 'model') and qa_model.model is not None,
        "vector_db_ready": getattr(vector_db, 'initialized', False)
    }


@app.get("/medical-disclaimer")
async def medical_disclaimer():
    return {
        "disclaimer": "This AI system provides general information only and should not be used for medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.",
        "warning": "Never disregard professional medical advice or delay seeking it because of something you read here."
    }


@app.post("/confidence-check")
async def confidence_check(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "")
        context = data.get("context", "")

        if not query or not context:
            raise HTTPException(status_code=400, detail="Query and context are required")

        confidence = qa_model.estimate_confidence(query, context)
        is_medical = is_medical_query(query)

        return {
            "confidence": confidence,
            "is_medical": is_medical,
            "min_confidence_recommended": 0.6 if is_medical else 0.4,
            "meets_threshold": confidence >= (0.6 if is_medical else 0.4)
        }

    except Exception as e:
        logger.error(f"Error in /confidence-check: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ------------------------------
# Run Server
# ------------------------------
if __name__ == "__main__":
    import uvicorn

    host = getattr(settings, "HOST", "127.0.0.1")
    port = getattr(settings, "PORT", 8000)
    debug = getattr(settings, "DEBUG", True)

    logger.info(f"🚀 FastAPI running at http://{host}:{port}")
    logger.info(f"📘 Swagger docs available at http://{host}:{port}/docs")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug
    )
