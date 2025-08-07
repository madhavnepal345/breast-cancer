from typing import Dict, List
from pydantic import BaseModel
import uuid
from datetime import datetime

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class Conversation:
    def __init__(self, session_id: str = None, max_history=5):
        self.session_id = session_id or str(uuid.uuid4())
        self.history: List[Message] = []
        self.max_history = max_history
        
    def add_message(self, role: str, content: str):
        self.history.append(
            Message(role=role, content=content, timestamp=datetime.now()))
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_formatted_history(self) -> str:
        return "\n".join(
            f"{msg.role}: {msg.content}" 
            for msg in self.history[-self.max_history:]
        )
    
    def resolve_references(self, query: str) -> str:
        """Handle pronouns and co-references"""
        # Simple implementation - replace "it" with last mentioned entity
        if "it" in query.lower().split():
            for msg in reversed(self.history):
                if "cancer" in msg.content.lower():
                    return query.replace("it", "breast cancer")
        return query

class ConversationManager:
    def __init__(self):
        self.sessions: Dict[str, Conversation] = {}
        
    def get_session(self, session_id: str) -> Conversation:
        if session_id not in self.sessions:
            self.sessions[session_id] = Conversation(session_id)
        return self.sessions[session_id]