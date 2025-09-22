"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Union


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    stream: bool = False


class QueryRequest(BaseModel):
    question: str
    max_results: int = 5
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    confidence_score: float
    response_type: str  # confident, clarification, greeting, not_found
    sources: List[Dict] = []
    suggestions: List[Union[str, Dict]] = []
    response_time: float


class SyncResponse(BaseModel):
    status: str
    message: str
    old_count: int
    new_count: int
    changes: Dict
    timestamp: str


class AddFAQRequest(BaseModel):
    question: str
    answer: str
    category: Optional[str] = "Umum"


class AddFAQResponse(BaseModel):
    status: str
    message: str
    faq_id: int
    question: str
    answer: str
    category: str
    timestamp: str


class UploadCSVResponse(BaseModel):
    status: str
    message: str
    total_processed: int
    successful_inserts: int
    failed_inserts: int
    errors: List[str]
    timestamp: str