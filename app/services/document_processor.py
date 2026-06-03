import httpx
import tempfile
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from app.core.config import settings

# ── Initialize clients ─────────────────────────────────────────────────────
def get_pinecone_index():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return pc.Index(settings.PINECONE_INDEX)

def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY
    )

# ── Download file from S3 URL ──────────────────────────────────────────────
async def download_file(url: str, suffix: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.close()
    return tmp.name

# ── Process document ───────────────────────────────────────────────────────
async def process_document(
    document_id: str,
    s3_url: str,
    file_type: str,
    org_id: str,
    department_id: str = None,
    visibility: str = "DEPARTMENT"
) -> int:
    """
    Downloads document from S3, splits into chunks,
    creates embeddings, stores in Pinecone.
    Returns number of chunks processed.
    """
    tmp_path = None
    try:
        # 1. Download file from S3
        suffix = f".{file_type}"
        tmp_path = await download_file(s3_url, suffix)

        # 2. Load and split document
        chunks = await load_and_split(tmp_path, file_type)

        if not chunks:
            raise ValueError("No text content found in document")

        # 3. Create embeddings and store in Pinecone
        await store_in_pinecone(
            chunks=chunks,
            document_id=document_id,
            org_id=org_id,
            department_id=department_id,
            visibility=visibility
        )

        return len(chunks)

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ── Load and split document into chunks ───────────────────────────────────
async def load_and_split(file_path: str, file_type: str) -> List:
    file_type = file_type.lower()

    # For PDFs
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        pages  = loader.load()
    # For text files
    elif file_type in ["txt", "csv"]:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8")
        pages  = loader.load()
    # For Word documents
    elif file_type in ["doc", "docx"]:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        pages  = loader.load()
    else:
        # Fallback — try as text
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        pages  = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # 1000 chars per chunk
        chunk_overlap=200,     # 200 chars overlap between chunks
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    return chunks

# ── Store chunks in Pinecone ───────────────────────────────────────────────
async def store_in_pinecone(
    chunks: List,
    document_id: str,
    org_id: str,
    department_id: str = None,
    visibility: str = "DEPARTMENT"
):
    embeddings_model = get_embeddings()
    index            = get_pinecone_index()

    # Delete existing vectors for this document (re-processing case)
    try:
        index.delete(filter={"document_id": document_id})
    except Exception:
        pass  # Index might be empty, that's fine

    # Batch process chunks
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch  = chunks[i:i + batch_size]
        texts  = [chunk.page_content for chunk in batch]

        # Create embeddings
        vectors = embeddings_model.embed_documents(texts)

        # Prepare records for Pinecone
        records = []
        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            records.append({
                "id": f"{document_id}_chunk_{i + j}",
                "values": vector,
                "metadata": {
                    "document_id":   document_id,
                    "org_id":        org_id,
                    "department_id": department_id or "",
                    "visibility":    visibility,
                    "text":          chunk.page_content,
                    "page":          chunk.metadata.get("page", 0),
                    "chunk_index":   i + j,
                }
            })

        # Upsert to Pinecone
        index.upsert(vectors=records)

# ── Delete document vectors from Pinecone ─────────────────────────────────
async def delete_document_vectors(document_id: str):
    index = get_pinecone_index()
    index.delete(filter={"document_id": document_id})