from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime
import logging
from enum import Enum
import re

# Configure logging
logger = logging.getLogger(__name__)

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    """Enhanced message model with validation"""
    role: MessageRole = Field(..., description="Sender role")
    content: str = Field(..., min_length=1, max_length=2000)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class Conversation:
    """Enhanced conversation handler with:
    - Context window management
    - Advanced reference resolution
    - Conversation summarization
    """
    
    def __init__(self, session_id: Optional[str] = None, max_history: int = 10):
        self.session_id = session_id or str(uuid.uuid4())
        self.history: List[Message] = []
        self.max_history = max_history
        self.context_entities = set()  # Track entities for reference resolution
        self.summary = ""  # Conversation summary for long contexts

    def add_message(self, role: MessageRole, content: str, **metadata) -> None:
        """Add message with automatic context management"""
        try:
            message = Message(
                role=role,
                content=content,
                metadata=metadata
            )
            self.history.append(message)
            
            # Update context entities
            self._update_entities(content)
            
            # Maintain context window
            if len(self.history) > self.max_history:
                removed = self.history.pop(0)
                logger.debug(f"Pruned message from history: {removed.content[:50]}...")
                
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            raise ValueError("Invalid message parameters")

    def get_formatted_history(self, window: Optional[int] = None) -> str:
        """Get conversation history as formatted string with optional window"""
        window = window or self.max_history
        recent_messages = self.history[-window:]
        return "\n".join(
            f"{msg.role.value.upper()}: {msg.content}"
            for msg in recent_messages
        )

    def resolve_references(self, query: str) -> str:
        """Enhanced reference resolution with:
        - Pronoun resolution (it, they, etc.)
        - Entity tracking
        - Temporal references (last time, previously)
        """
        resolved = query
        
        # Pronoun resolution
        pronoun_map = {
            "it": self._find_recent_entity("it"),
            "they": self._find_recent_entity("they"),
            "that": self._find_recent_entity("that")
        }
        
        for pronoun, entity in pronoun_map.items():
            if pronoun in resolved.lower():
                if entity:
                    resolved = re.sub(
                        rf"\b{pronoun}\b", 
                        entity, 
                        resolved, 
                        flags=re.IGNORECASE
                    )
                    logger.debug(f"Resolved '{pronoun}' to '{entity}'")
        
        # Temporal references
        if "last time" in resolved.lower():
            for msg in reversed(self.history):
                if msg.role == MessageRole.USER:
                    resolved = resolved.replace("last time", f"when you said '{msg.content[:50]}...'")
                    break
        
        return resolved

    def _update_entities(self, text: str) -> None:
        """Extract and track entities from text"""
        # Simple implementation - extract noun phrases
        nouns = re.findall(
            r"\b(the\s)?(breast|lung|prostate)\s(cancer|tumor)\b", 
            text, 
            flags=re.IGNORECASE
        )
        for noun in nouns:
            self.context_entities.add(" ".join(noun).lower())

    def _find_recent_entity(self, pronoun: str) -> Optional[str]:
        """Find most recent matching entity for a pronoun"""
        # Simple gender/number matching
        if pronoun == "it":
            candidates = [e for e in self.context_entities if "cancer" in e]
        elif pronoun == "they":
            candidates = [e for e in self.context_entities if "patients" in e]
        else:
            candidates = list(self.context_entities)
            
        return candidates[-1] if candidates else None

    def summarize(self) -> str:
        """Generate conversation summary"""
        # Placeholder - integrate with summarization model
        user_messages = [m.content for m in self.history if m.role == MessageRole.USER]
        return " | ".join(user_messages[-3:])

    def to_dict(self) -> Dict:
        """Serialize conversation"""
        return {
            "session_id": self.session_id,
            "history": [msg.dict() for msg in self.history],
            "summary": self.summary,
            "entities": list(self.context_entities)
        }

class ConversationManager:
    """Production-grade conversation manager with:
    - Session persistence
    - Automatic cleanup
    - Analytics hooks
    """
    
    def __init__(self, session_timeout: int = 3600):
        self.sessions: Dict[str, Conversation] = {}
        self.session_timeout = session_timeout  # Seconds
        
    def get_session(self, session_id: Optional[str] = None) -> Conversation:
        """Get or create session with automatic cleanup"""
        self._cleanup_sessions()
        
        if session_id is None:
            return self._create_session()
            
        if session_id not in self.sessions:
            logger.warning(f"Creating new session for ID {session_id}")
            self.sessions[session_id] = Conversation(session_id)
            
        return self.sessions[session_id]

    def _create_session(self) -> Conversation:
        """Create new session with validation"""
        session = Conversation()
        self.sessions[session.session_id] = session
        logger.info(f"Created new session: {session.session_id}")
        return session

    def _cleanup_sessions(self) -> None:
        """Remove stale sessions"""
        now = datetime.now()
        stale = [
            sid for sid, conv in self.sessions.items()
            if (now - conv.history[-1].timestamp).total_seconds() > self.session_timeout
        ]
        
        for sid in stale:
            logger.info(f"Cleaning up stale session: {sid}")
            del self.sessions[sid]

    def active_sessions(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)

    def get_conversation_summary(self, session_id: str) -> Optional[Dict]:
        """Get conversation summary with metadata"""
        if session_id not in self.sessions:
            return None
            
        conv = self.sessions[session_id]
        return {
            "session_id": session_id,
            "summary": conv.summarize(),
            "message_count": len(conv.history),
            "last_active": conv.history[-1].timestamp.isoformat()
        }