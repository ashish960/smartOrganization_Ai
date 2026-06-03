from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from app.core.config import settings
from app.models.schemas import ChatMessage

# ── Initialize clients ─────────────────────────────────────────────────────
def get_pinecone_index():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return pc.Index(settings.PINECONE_INDEX)

def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY
    )

def get_llm():
    return ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0.1,       # low temperature = more factual answers
        max_tokens=1000,
    )

# ── Build Pinecone filter based on user's access ───────────────────────────
def build_access_filter(
    org_id: str,
    department_id: Optional[str],
    allowed_dept_ids: List[str],
    role: str
) -> dict:
    """
    Builds a Pinecone metadata filter so users only see
    documents they have access to based on their role and department.
    """
    if role == "ORG_ADMIN":
        # ORG_ADMIN sees everything in their org
        return {"org_id": org_id}

    elif role == "VIEWER":
        # VIEWER sees only PUBLIC docs and their own dept docs
        if department_id:
            return {
                "org_id": org_id,
                "$or": [
                    {"visibility": "PUBLIC"},
                    {"department_id": department_id},
                ]
            }
        return {"org_id": org_id, "visibility": "PUBLIC"}

    else:
        # USER / DEPT_MANAGER — public + own dept + cross-access depts
        accessible_depts = list(set([department_id] + (allowed_dept_ids or []))) if department_id else allowed_dept_ids or []

        if accessible_depts:
            return {
                "org_id": org_id,
                "$or": [
                    {"visibility": "PUBLIC"},
                    *[{"department_id": dept_id} for dept_id in accessible_depts],
                ]
            }
        return {"org_id": org_id, "visibility": "PUBLIC"}

# ── Retrieve relevant chunks from Pinecone ────────────────────────────────
async def retrieve_relevant_chunks(
    question: str,
    org_id: str,
    department_id: Optional[str],
    allowed_dept_ids: List[str],
    role: str,
    top_k: int = 5
) -> List[dict]:
    embeddings_model = get_embeddings()
    index            = get_pinecone_index()

    # 1. Embed the question
    question_vector = embeddings_model.embed_query(question)

    # 2. Build access filter
    access_filter = build_access_filter(org_id, department_id, allowed_dept_ids, role)

    # 3. Search Pinecone
    results = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True,
        filter=access_filter
    )

    # 4. Return relevant chunks
    chunks = []
    for match in results.matches:
        if match.score > 0.7:  # only include high-relevance chunks
            chunks.append({
                "text":        match.metadata.get("text", ""),
                "document_id": match.metadata.get("document_id", ""),
                "page":        match.metadata.get("page", 0),
                "score":       round(match.score, 3),
            })

    return chunks

# ── Generate answer using GPT ─────────────────────────────────────────────
async def generate_answer(
    question: str,
    context_chunks: List[dict],
    conversation_history: List[ChatMessage]
) -> str:
    llm    = get_llm()
    parser = StrOutputParser()

    # Build context from retrieved chunks
    if context_chunks:
        context = "\n\n".join([
            f"[Document chunk {i+1}]:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
    else:
        context = "No relevant documents found."

    # Build conversation history string
    history_text = ""
    if conversation_history:
        for msg in conversation_history[-6:]:  # last 6 messages for context
            role = "User" if msg.role == "user" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

    # System prompt
    system_prompt = """You are SmartOrg AI, an intelligent document assistant for enterprise organizations.
Your job is to answer questions based ONLY on the provided document context.

Rules:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "I couldn't find relevant information in your documents for this question"
- Be concise and professional
- Cite which document chunk you used when relevant
- Never make up information not present in the context"""

    # Build the prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """Previous conversation:
{history}

Document context:
{context}

Current question: {question}

Please answer based on the document context provided.""")
    ])

    chain = prompt_template | llm | parser

    answer = await chain.ainvoke({
        "history":  history_text,
        "context":  context,
        "question": question,
    })

    return answer

# ── Main chat function ────────────────────────────────────────────────────
async def chat(
    question: str,
    org_id: str,
    department_id: Optional[str],
    allowed_dept_ids: List[str],
    conversation_history: List[ChatMessage],
    user_id: str,
    role: str,
) -> dict:
    # 1. Retrieve relevant chunks from Pinecone
    chunks = await retrieve_relevant_chunks(
        question=question,
        org_id=org_id,
        department_id=department_id,
        allowed_dept_ids=allowed_dept_ids,
        role=role,
    )

    # 2. Generate answer using GPT
    answer = await generate_answer(
        question=question,
        context_chunks=chunks,
        conversation_history=conversation_history,
    )

    # 3. Update conversation history
    updated_history = conversation_history + [
        ChatMessage(role="user",      content=question),
        ChatMessage(role="assistant", content=answer),
    ]

    # 4. Build sources (without duplicate documents)
    seen_docs = set()
    sources   = []
    for chunk in chunks:
        doc_id = chunk["document_id"]
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "document_id": doc_id,
                "page":        chunk["page"],
                "relevance":   chunk["score"],
            })

    return {
        "answer":               answer,
        "sources":              sources,
        "conversation_history": updated_history,
    }