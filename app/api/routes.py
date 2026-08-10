from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.models.schemas import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
)
from app.services.document_processor import process_document, delete_document_vectors
from app.services.chat_service import chat

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="SmartOrg AI Service",
    )


@router.post("/process-document", response_model=ProcessDocumentResponse)
async def process_document_route(
    request: ProcessDocumentRequest,
    x_internal_key: Optional[str] = Header(None),
):
    from app.core.config import settings

    if x_internal_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        chunks_count = await process_document(
            document_id=request.document_id,
            s3_url=request.s3_url,
            file_type=request.file_type,
            org_id=request.org_id,
            department_id=request.department_id,
            visibility=request.visibility,
        )

        return ProcessDocumentResponse(
            success=True,
            document_id=request.document_id,
            chunks_processed=chunks_count,
            message=f"Document processed successfully into {chunks_count} chunks",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/document/{document_id}")
async def delete_document_route(
    document_id: str,
    x_internal_key: Optional[str] = Header(None),
):
    from app.core.config import settings

    if x_internal_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        await delete_document_vectors(document_id)
        return {"success": True, "message": "Document vectors deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_route(
    request: ChatRequest,
    x_internal_key: Optional[str] = Header(None),
):
    from app.core.config import settings

    if x_internal_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        result = await chat(
            question=request.question,
            org_id=request.org_id,
            department_id=request.department_id,
            allowed_dept_ids=request.allowed_dept_ids or [],
            conversation_history=request.conversation_history or [],
            user_id=request.user_id,
            role=request.role,
            scope_type=request.scope_type,
            scope_document_id=request.scope_document_id,
            scope_department_id=request.scope_department_id,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_history=result["conversation_history"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
