import httpx
import tempfile
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from app.core.config import settings


def get_pinecone_index():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return pc.Index(settings.PINECONE_INDEX)


def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
    )


async def download_file(url: str, suffix: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.close()
    return tmp.name


async def process_document(
    document_id: str,
    s3_url: str,
    file_type: str,
    org_id: str,
    department_id: str = None,
    visibility: str = "DEPARTMENT",
) -> int:
    tmp_path = None
    try:
        suffix = f".{file_type}"
        tmp_path = await download_file(s3_url, suffix)

        chunks = await load_and_split(tmp_path, file_type)

        if not chunks:
            raise ValueError("No text content found in document")

        await store_in_pinecone(
            chunks=chunks,
            document_id=document_id,
            org_id=org_id,
            department_id=department_id,
            visibility=visibility,
        )

        return len(chunks)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def load_and_split(file_path: str, file_type: str) -> List:
    file_type = file_type.lower()

    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    elif file_type in ["txt", "csv"]:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8")
        pages = loader.load()
    elif file_type in ["doc", "docx"]:
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(file_path)
        pages = loader.load()
    else:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    return chunks


async def store_in_pinecone(
    chunks: List,
    document_id: str,
    org_id: str,
    department_id: str = None,
    visibility: str = "DEPARTMENT",
):
    embeddings_model = get_embeddings()
    index = get_pinecone_index()

    try:
        index.delete(filter={"document_id": document_id})
    except Exception:
        pass

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.page_content for chunk in batch]

        vectors = embeddings_model.embed_documents(texts)

        records = []
        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            records.append(
                {
                    "id": f"{document_id}_chunk_{i + j}",
                    "values": vector,
                    "metadata": {
                        "document_id": document_id,
                        "org_id": org_id,
                        "department_id": department_id if department_id else "NO_DEPT",
                        "visibility": visibility,
                        "text": chunk.page_content,
                        "page": chunk.metadata.get("page", 0),
                        "chunk_index": i + j,
                    },
                }
            )

        index.upsert(vectors=records)


async def delete_document_vectors(document_id: str):
    index = get_pinecone_index()
    index.delete(filter={"document_id": document_id})
