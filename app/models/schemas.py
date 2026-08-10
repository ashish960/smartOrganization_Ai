from pydantic import BaseModel
from typing import Optional, List


class ProcessDocumentRequest(BaseModel):
    document_id: str
    s3_url: str
    file_type: str
    org_id: str
    department_id: Optional[str] = None
    visibility: str = "DEPARTMENT"


class ProcessDocumentResponse(BaseModel):
    success: bool
    document_id: str
    chunks_processed: int
    message: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    org_id: str
    department_id: Optional[str] = None
    allowed_dept_ids: Optional[List[str]] = []
    conversation_history: Optional[List[ChatMessage]] = []
    user_id: str
    role: str
    scope_type: str = "all"
    scope_document_id: Optional[str] = None
    scope_department_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    conversation_history: List[ChatMessage]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str = "1.0.0"
